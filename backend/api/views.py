from rest_framework.decorators import api_view, parser_classes
from rest_framework.response import Response
from rest_framework import status, viewsets
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.decorators import action
from django.http import FileResponse, Http404, HttpResponse
from django.shortcuts import get_object_or_404
from django.utils import timezone
from django.db import models as dj_models
import logging
import os
import threading
import fitz
import io
import base64

from .models import Document, ExtractedFundData, DocumentChangeLog
from .serializers import (
    MessageSerializer,
    DocumentSerializer,
    DocumentUploadSerializer,
    DocumentListSerializer,
    ExtractedFundDataSerializer,
    DocumentChangeLogSerializer,
    ChatRequestSerializer,
    ChatResponseSerializer,
    ChatHistorySerializer
)
from .services import DocumentProcessingService, RAGService

logger = logging.getLogger(__name__)


class DocumentViewSet(viewsets.ModelViewSet):
    """
    ViewSet for handling document CRUD operations
    """
    queryset = Document.objects.all()
    serializer_class = DocumentSerializer
    
    def get_serializer_class(self):
        """Return appropriate serializer based on action"""
        if self.action == 'list':
            return DocumentListSerializer
        elif self.action == 'create':
            return DocumentUploadSerializer
        return DocumentSerializer
    
    def list(self, request, *args, **kwargs):
        """List all documents with basic info"""
        queryset = self.get_queryset()
        serializer = self.get_serializer(queryset, many=True)
        return Response({
            'count': queryset.count(),
            'results': serializer.data
        })
    
    def create(self, request, *args, **kwargs):
        """
        Upload a new document and trigger processing
        """
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        # Save document
        document = serializer.save()

        # Start RAG ingestion immediately after upload (runs in background)
        # This overlaps the 5-7 minute chunking/embedding time with OCR extraction.
        auto_rag_raw = os.getenv("AUTO_RAG_INGEST_ON_UPLOAD", "true").strip().lower()
        auto_rag_enabled = auto_rag_raw not in {"0", "false", "no", "off"}
        if auto_rag_enabled:
            try:
                # Mark queued so the UI can show progress right away
                Document.objects.filter(id=document.id).update(
                    rag_status='queued',
                    rag_progress=0,
                    rag_error_message=None,
                    rag_started_at=None,
                    rag_completed_at=None,
                )
            except Exception:
                pass

            def _rag_task(doc_id: int):
                try:
                    from django.db import close_old_connections

                    close_old_connections()
                    doc = Document.objects.get(id=doc_id)

                    # Avoid duplicate ingestion
                    if getattr(doc, 'rag_status', None) in {'running', 'completed'} or doc.chunks.exists():
                        return

                    logger.info(f"AUTO_RAG_INGEST_ON_UPLOAD: starting RAG ingestion for document {doc_id}")
                    RAGService().ingest_document(doc_id)
                    close_old_connections()
                except Exception as e:
                    logger.error(f"AUTO_RAG_INGEST_ON_UPLOAD: RAG ingestion failed for document {doc_id}: {str(e)}")

            threading.Thread(target=_rag_task, args=(document.id,), daemon=True).start()
        
        # Start async processing
        try:
            processing_service = DocumentProcessingService()
            processing_service.process_document(document.id)
            logger.info(f"Started processing for document {document.id}")

            # RAG ingestion now starts automatically after processing completes
            # (see DocumentProcessingService._process_document_task).
        except Exception as e:
            logger.error(f"Failed to start processing: {str(e)}")
            document.status = 'failed'
            document.error_message = f"Failed to start processing: {str(e)}"
            document.save()
        
        # Return response with document details
        response_serializer = DocumentSerializer(document, context={'request': request})
        return Response(
            response_serializer.data,
            status=status.HTTP_201_CREATED
        )
    
    def retrieve(self, request, *args, **kwargs):
        """Get detailed information about a specific document"""
        instance = self.get_object()
        serializer = DocumentSerializer(instance, context={'request': request})
        return Response(serializer.data)
    
    def update(self, request, *args, **kwargs):
        """Update document - supports updating extracted_data"""
        partial = kwargs.pop('partial', False)
        instance = self.get_object()
        
        # Store old data for change tracking
        old_extracted_data = instance.extracted_data.copy() if instance.extracted_data else {}
        user_comment = request.data.get('user_comment', '')
        
        serializer = self.get_serializer(instance, data=request.data, partial=partial)
        serializer.is_valid(raise_exception=True)
        
        # If extracted_data was updated, increment edit count and track changes
        if 'extracted_data' in request.data:
            new_extracted_data = request.data['extracted_data']
            
            # Detect changes
            changes = self._detect_changes(old_extracted_data, new_extracted_data)
            
            if changes:
                instance.edit_count += 1
                instance.last_edited_at = timezone.now()
                
                # Create change log entry
                DocumentChangeLog.objects.create(
                    document=instance,
                    user_comment=user_comment,
                    changes=changes
                )
        
        self.perform_update(serializer)
        
        # If extracted_data was updated, sync with ExtractedFundData
        if 'extracted_data' in request.data:
            try:
                from .services import DocumentProcessingService
                # Re-sync the structured data
                extracted_data = instance.extracted_data
                
                # Helper to get value from structured or flat format
                def get_value(field_data):
                    if isinstance(field_data, dict) and 'value' in field_data:
                        return field_data['value']
                    return field_data

                def truncate_defaults_for_model(model_cls, defaults: dict) -> dict:
                    """Ensure values fit DB column limits (e.g., CharField max_length)."""
                    sanitized = dict(defaults)
                    for field_name, field_value in sanitized.items():
                        if not isinstance(field_value, str) or field_value is None:
                            continue
                        try:
                            model_field = model_cls._meta.get_field(field_name)
                        except Exception:
                            continue
                        if not isinstance(model_field, dj_models.CharField):
                            continue
                        max_len = getattr(model_field, 'max_length', None)
                        if max_len and len(field_value) > max_len:
                            logger.warning(
                                f"Truncating {model_cls.__name__}.{field_name}: {len(field_value)} -> {max_len} chars"
                            )
                            sanitized[field_name] = field_value[:max_len]
                    return sanitized
                
                # Update ExtractedFundData
                fund_defaults = {
                        'fund_name': get_value(extracted_data.get('fund_name')),
                        'fund_code': get_value(extracted_data.get('fund_code')),
                        'fund_type': get_value(extracted_data.get('fund_type')),
                        'legal_structure': get_value(extracted_data.get('legal_structure')),
                        'license_number': get_value(extracted_data.get('license_number')),
                        'regulator': get_value(extracted_data.get('regulator')),
                        'management_company': get_value(extracted_data.get('management_company')),
                        'custodian_bank': get_value(extracted_data.get('custodian_bank')),
                        'fund_supervisor': get_value(extracted_data.get('fund_supervisor')),
                        'management_fee': str(get_value(extracted_data.get('fees', {}).get('management_fee'))) if extracted_data.get('fees', {}).get('management_fee') else None,
                        'subscription_fee': str(get_value(extracted_data.get('fees', {}).get('subscription_fee'))) if extracted_data.get('fees', {}).get('subscription_fee') else None,
                        'redemption_fee': str(get_value(extracted_data.get('fees', {}).get('redemption_fee'))) if extracted_data.get('fees', {}).get('redemption_fee') else None,
                        'switching_fee': str(get_value(extracted_data.get('fees', {}).get('switching_fee'))) if extracted_data.get('fees', {}).get('switching_fee') else None,
                        'total_expense_ratio': str(get_value(extracted_data.get('fees', {}).get('total_expense_ratio'))) if extracted_data.get('fees', {}).get('total_expense_ratio') else None,
                        'custody_fee': str(get_value(extracted_data.get('fees', {}).get('custody_fee'))) if extracted_data.get('fees', {}).get('custody_fee') else None,
                        'audit_fee': str(get_value(extracted_data.get('fees', {}).get('audit_fee'))) if extracted_data.get('fees', {}).get('audit_fee') else None,
                        'supervisory_fee': str(get_value(extracted_data.get('fees', {}).get('supervisory_fee'))) if extracted_data.get('fees', {}).get('supervisory_fee') else None,
                        'other_expenses': str(get_value(extracted_data.get('fees', {}).get('other_expenses'))) if extracted_data.get('fees', {}).get('other_expenses') else None,

                        'investment_objective': get_value(extracted_data.get('investment_objective')),
                        'investment_strategy': get_value(extracted_data.get('investment_strategy')),
                        'investment_style': get_value(extracted_data.get('investment_style')),
                        'sector_focus': get_value(extracted_data.get('sector_focus')),
                        'benchmark': get_value(extracted_data.get('benchmark')),

                        'valuation_method': get_value(extracted_data.get('valuation', {}).get('valuation_method')) if extracted_data.get('valuation') else get_value(extracted_data.get('valuation_method')),
                        'pricing_source': get_value(extracted_data.get('valuation', {}).get('pricing_source')) if extracted_data.get('valuation') else get_value(extracted_data.get('pricing_source')),

                        'investment_restrictions': get_value(extracted_data.get('investment_restrictions')),
                        'borrowing_limit': get_value(extracted_data.get('borrowing_limit')),
                        'leverage_limit': get_value(extracted_data.get('leverage_limit')),

                        'investor_rights': get_value(extracted_data.get('investor_rights')),
                        'distribution_agent': get_value(extracted_data.get('distribution_agent')),
                        'sales_channels': get_value(extracted_data.get('sales_channels')),

                        'concentration_risk': get_value(extracted_data.get('risk_factors', {}).get('concentration_risk')) if extracted_data.get('risk_factors') else get_value(extracted_data.get('concentration_risk')),
                        'liquidity_risk': get_value(extracted_data.get('risk_factors', {}).get('liquidity_risk')) if extracted_data.get('risk_factors') else get_value(extracted_data.get('liquidity_risk')),
                        'interest_rate_risk': get_value(extracted_data.get('risk_factors', {}).get('interest_rate_risk')) if extracted_data.get('risk_factors') else get_value(extracted_data.get('interest_rate_risk')),

                        'trading_frequency': get_value(extracted_data.get('operational_details', {}).get('trading_frequency')) if extracted_data.get('operational_details') else get_value(extracted_data.get('trading_frequency')),
                        'cut_off_time': get_value(extracted_data.get('operational_details', {}).get('cut_off_time')) if extracted_data.get('operational_details') else get_value(extracted_data.get('cut_off_time')),
                        'nav_calculation_frequency': get_value(extracted_data.get('operational_details', {}).get('nav_calculation_frequency')) if extracted_data.get('operational_details') else get_value(extracted_data.get('nav_calculation_frequency')),
                        'nav_publication': get_value(extracted_data.get('operational_details', {}).get('nav_publication')) if extracted_data.get('operational_details') else get_value(extracted_data.get('nav_publication')),
                        'settlement_cycle': get_value(extracted_data.get('operational_details', {}).get('settlement_cycle')) if extracted_data.get('operational_details') else get_value(extracted_data.get('settlement_cycle')),

                        'auditor': get_value(extracted_data.get('governance', {}).get('auditor')) if extracted_data.get('governance') else get_value(extracted_data.get('auditor')),

                        'asset_allocation': extracted_data.get('asset_allocation') if isinstance(extracted_data.get('asset_allocation'), dict) else {},
                        'minimum_investment': extracted_data.get('minimum_investment') if isinstance(extracted_data.get('minimum_investment'), dict) else {},
                        'portfolio': extracted_data.get('portfolio', []),
                        'nav_history': extracted_data.get('nav_history', []),
                        'dividend_history': extracted_data.get('dividend_history', []),

                }

                fund_defaults = truncate_defaults_for_model(ExtractedFundData, fund_defaults)

                ExtractedFundData.objects.update_or_create(
                    document=instance,
                    defaults=fund_defaults,
                )
                
                logger.info(f"Document {instance.id} edited. Total edits: {instance.edit_count}")
            except Exception as e:
                logger.error(f"Error syncing ExtractedFundData: {e}")
        
        return Response(serializer.data)
    
    def _detect_changes(self, old_data, new_data, prefix=''):
        """Recursively detect changes between old and new data"""
        changes = {}
        
        # Helper to get value from structured or flat format
        def get_value(field_data):
            if isinstance(field_data, dict) and 'value' in field_data:
                return field_data['value']
            return field_data
        
        # Get all keys from both old and new data
        all_keys = set(old_data.keys()) | set(new_data.keys())
        
        for key in all_keys:
            field_path = f"{prefix}.{key}" if prefix else key
            old_val = old_data.get(key)
            new_val = new_data.get(key)
            
            # Handle nested dictionaries (but not structured fields with 'value' key)
            if isinstance(old_val, dict) and isinstance(new_val, dict):
                # Check if it's a structured field (has 'value' key)
                if 'value' in old_val or 'value' in new_val:
                    old_value = get_value(old_val)
                    new_value = get_value(new_val)
                    if old_value != new_value:
                        changes[field_path] = {'old': old_value, 'new': new_value}
                else:
                    # Recursively check nested dict
                    nested_changes = self._detect_changes(old_val, new_val, field_path)
                    changes.update(nested_changes)
            else:
                # Direct comparison for simple values
                old_value = get_value(old_val) if old_val else old_val
                new_value = get_value(new_val) if new_val else new_val
                
                if old_value != new_value:
                    changes[field_path] = {'old': old_value, 'new': new_value}
        
        return changes
    
    def partial_update(self, request, *args, **kwargs):
        """Partial update (PATCH)"""
        kwargs['partial'] = True
        return self.update(request, *args, **kwargs)
    
    @action(detail=True, methods=['post'])
    def reprocess(self, request, pk=None):
        """
        Reprocess a document
        """
        document = self.get_object()
        
        # Check if document can be reprocessed
        if document.status == 'processing':
            return Response(
                {'error': 'Document is already being processed'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Reset status and trigger processing
        document.status = 'pending'
        document.error_message = None
        document.extracted_data = None
        document.save()
        
        # Start processing
        processing_service = DocumentProcessingService()
        processing_service.process_document(document.id)

        # Optional: also auto-ingest for RAG on reprocess
        auto_rag = os.getenv("AUTO_RAG_INGEST_ON_UPLOAD", "").strip().lower() in {"1", "true", "yes"}
        if auto_rag:
            def _rag_task(doc_id: int):
                try:
                    logger.info(f"AUTO_RAG_INGEST_ON_UPLOAD enabled. Starting RAG ingestion for document {doc_id}")
                    RAGService().ingest_document(doc_id)
                    logger.info(f"Auto RAG ingestion completed for document {doc_id}")
                except Exception as e:
                    logger.error(f"Auto RAG ingestion failed for document {doc_id}: {str(e)}")

            threading.Thread(target=_rag_task, args=(document.id,), daemon=True).start()
        
        serializer = DocumentSerializer(document, context={'request': request})
        return Response(serializer.data)
    
    @action(detail=True, methods=['get'])
    def optimized_pages(self, request, pk=None):
        """
        Get all pages from the optimized PDF as images (base64)
        """
        document = self.get_object()
        
        if not document.optimized_file:
            return Response(
                {'error': 'No optimized PDF available for this document'},
                status=status.HTTP_404_NOT_FOUND
            )
        
        try:
            # Get the page map (optimized index -> raw page number)
            page_map = None
            if isinstance(document.extracted_data, dict):
                page_map = document.extracted_data.get('_optimized_page_map')
            
            doc = fitz.open(document.optimized_file.path)
            pages = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                # Render page to image at 150 DPI for preview
                pix = page.get_pixmap(dpi=150)
                img_bytes = pix.tobytes("png")
                
                # Convert to base64 for JSON transmission
                img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                
                # Get raw page number from map if available
                raw_page_num = page_map[page_num] if isinstance(page_map, list) and page_num < len(page_map) else page_num + 1
                
                pages.append({
                    'page_number': page_num + 1,
                    'raw_page_number': raw_page_num,
                    'image': f'data:image/png;base64,{img_base64}',
                    'width': pix.width,
                    'height': pix.height
                })
            
            doc.close()
            
            return Response({
                'total_pages': len(pages),
                'pages': pages
            })
            
        except Exception as e:
            logger.error(f"Error extracting optimized pages: {str(e)}")
            return Response(
                {'error': f'Failed to extract pages: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    @action(detail=True, methods=['get'], url_path='preview-page/(?P<page_num>[0-9]+)')
    def preview_page(self, request, pk=None, page_num=None):
        """
        Returns an image of a specific page with highlights burned in.
        GET /api/documents/{id}/preview-page/{page_num}/
        
        page_num is the RAW page number from the original PDF.
        This endpoint maps it to the optimized page for rendering.
        """
        document = self.get_object()
        raw_page_num = int(page_num)
        
        # 1. Gather all bboxes for this RAW page from extracted_data
        bboxes_to_draw = []
        data = document.extracted_data or {}
        
        # Helper recursive function to find bboxes in nested JSON
        def find_bboxes(obj, label_prefix=''):
            if isinstance(obj, dict):
                # Check if this object is a field with bbox info
                if 'bbox' in obj and 'page' in obj:
                    if obj['page'] == raw_page_num:
                        bboxes_to_draw.append({
                            'bbox': obj['bbox'],
                            'label': label_prefix or 'Field',
                            'value': obj.get('value')
                        })
                # Continue searching deeper
                for k, v in obj.items():
                    if k not in ['bbox', 'page', 'value']:
                        new_label = k.replace('_', ' ').title()
                        find_bboxes(v, label_prefix=new_label)
            elif isinstance(obj, list):
                for idx, item in enumerate(obj):
                    find_bboxes(item, label_prefix=f"{label_prefix} {idx+1}" if label_prefix else f"Item {idx+1}")

        find_bboxes(data)
        
        logger.info(f"Found {len(bboxes_to_draw)} bboxes to draw on raw page {raw_page_num}")

        # 2. Map raw page number to optimized page number if we have an optimized file
        page_map = data.get('_optimized_page_map') if isinstance(data, dict) else None
        optimized_page_num = None
        
        if document.optimized_file and isinstance(page_map, list):
            try:
                idx = page_map.index(raw_page_num)
                optimized_page_num = idx + 1  # Convert 0-based index to 1-based page
            except ValueError:
                optimized_page_num = None
        
        # Determine which PDF to use and which page number
        if document.optimized_file and optimized_page_num is not None:
            pdf_path = document.optimized_file.path
            render_page_num = optimized_page_num
        else:
            pdf_path = document.file.path
            render_page_num = raw_page_num
        
        from .services import GeminiOCRService
        service = GeminiOCRService()
        image_path = service.generate_annotated_image(pdf_path, render_page_num, bboxes_to_draw)
        
        if not image_path or not os.path.exists(image_path):
            return Response(
                {"error": "Could not generate preview"}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

        # 3. Return the image as a file response
        return FileResponse(open(image_path, 'rb'), content_type='image/png')

    @action(detail=True, methods=['get'])
    def change_logs(self, request, pk=None):
        """
        Get all change logs for a document
        """
        document = self.get_object()
        logs = document.change_logs.all()
        serializer = DocumentChangeLogSerializer(logs, many=True)
        return Response({
            'count': logs.count(),
            'results': serializer.data
        })

    @action(detail=True, methods=['get'])
    def download(self, request, pk=None):
        """
        Download the original PDF file
        """
        document = self.get_object()
        
        if not document.file:
            raise Http404("File not found")
        
        try:
            return FileResponse(
                document.file.open('rb'),
                as_attachment=True,
                filename=document.file_name
            )
        except Exception as e:
            logger.error(f"Error downloading file: {str(e)}")
            raise Http404("File not found")
    
    @action(detail=False, methods=['get'])
    def stats(self, request):
        """
        Get processing statistics
        """
        total = Document.objects.count()
        pending = Document.objects.filter(status='pending').count()
        processing = Document.objects.filter(status='processing').count()
        completed = Document.objects.filter(status='completed').count()
        failed = Document.objects.filter(status='failed').count()
        
        return Response({
            'total': total,
            'pending': pending,
            'processing': processing,
            'completed': completed,
            'failed': failed
        })

    @action(detail=True, methods=['get'])
    def rag_status(self, request, pk=None):
        """
        Check if document is already ingested for RAG
        GET /api/documents/{id}/rag_status/
        """
        document = self.get_object()
        chunks_count = document.chunks.count()
        
        return Response({
            'is_ingested': chunks_count > 0,
            'chunks_count': chunks_count,
            'document_id': document.id,
            'rag_status': getattr(document, 'rag_status', None),
            'rag_progress': getattr(document, 'rag_progress', None),
            'rag_error_message': getattr(document, 'rag_error_message', None),
            'rag_started_at': getattr(document, 'rag_started_at', None),
            'rag_completed_at': getattr(document, 'rag_completed_at', None),
        })
    
    @action(detail=True, methods=['post'])
    def ingest_for_rag(self, request, pk=None):
        """
        Process document for RAG (vectorize and store chunks)
        POST /api/documents/{id}/ingest_for_rag/
        """
        document = self.get_object()
        
        if document.status != 'completed':
            return Response(
                {'error': 'Document must be processed successfully before RAG ingestion'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        try:
            logger.info(f"Starting RAG ingestion for document {document.id}")
            rag_service = RAGService()
            success = rag_service.ingest_document(document.id)
            
            chunks_count = document.chunks.count()
            logger.info(f"RAG ingestion completed. Created {chunks_count} chunks.")
            
            return Response({
                'message': 'Document ingested successfully for RAG',
                'chunks_count': chunks_count,
                'document_id': document.id
            })
        except Exception as e:
            logger.error(f"RAG ingestion error for document {document.id}: {str(e)}")

            # Treat common ingestion failures as client-visible 400s (missing files / no extractable text).
            msg = str(e)
            is_expected = isinstance(e, ValueError) and (
                'not found on disk' in msg
                or 'No PDF file found on disk' in msg
                or 'Could not extract text content' in msg
                or 'Extracted text is empty' in msg
            )
            http_status = status.HTTP_400_BAD_REQUEST if is_expected else status.HTTP_500_INTERNAL_SERVER_ERROR
            return Response(
                {'error': f'Failed to ingest document: {str(e)}'},
                status=http_status
            )

    @action(detail=True, methods=['post'])
    def chat(self, request, pk=None):
        """
        Chat with document using RAG
        POST /api/documents/{id}/chat/
        Body: {"query": "What is the management fee?", "history": [...]}
        """
        document = self.get_object()
        
        # Check if document has been ingested
        if not document.chunks.exists():
            return Response(
                {
                    'error': 'Document not ingested yet for RAG. Please call /documents/{id}/ingest_for_rag/ first.',
                    'chunks_count': 0
                },
                status=status.HTTP_400_BAD_REQUEST
            )
        
        # Validate request data
        serializer = ChatRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        user_query = serializer.validated_data['query']
        history = serializer.validated_data.get('history', [])
        
        try:
            logger.info(f"RAG chat query for document {document.id}: {user_query[:50]}...")
            rag_service = RAGService()
            answer_payload = rag_service.chat(document.id, user_query, history, return_source=True)

            answer_text = answer_payload.get('text') if isinstance(answer_payload, dict) else str(answer_payload)
            citations = answer_payload.get('citations', []) if isinstance(answer_payload, dict) else []

            response_data = {
                'answer': answer_text,
                'query': user_query,
                'chunks_count': document.chunks.count(),
                'citations': citations,
            }
            
            return Response(response_data)
        except Exception as e:
            logger.error(f"RAG chat error for document {document.id}: {str(e)}")
            return Response(
                {'error': f'Failed to process chat: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @action(detail=True, methods=['get', 'put'], url_path='chat_history')
    def chat_history(self, request, pk=None):
        """Persist / restore chat history for a document.

        GET  /api/documents/{id}/chat_history/ -> {"history": [...]}
        PUT  /api/documents/{id}/chat_history/ with body {"history": [...]} to overwrite.
        """
        document = self.get_object()

        if request.method.lower() == 'get':
            return Response({'history': document.chat_history or []})

        serializer = ChatHistorySerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        history = serializer.validated_data.get('history', [])
        # Keep it bounded to avoid unbounded DB growth.
        max_messages = 200
        if isinstance(history, list) and len(history) > max_messages:
            history = history[-max_messages:]

        document.chat_history = history
        document.save(update_fields=['chat_history'])
        return Response({'history': document.chat_history or []})

    @action(detail=True, methods=['get'], url_path='page-context/(?P<page_num>[0-9]+)')
    def page_context(self, request, pk=None, page_num=None):
        """
        Returns a page image + bbox list for any quoted text on that page.
        GET /api/documents/{id}/page-context/{page_num}/?quote=...
        page_num is the RAW (original) 1-based page number.
        """
        document = self.get_object()
        raw_page_num = int(page_num)
        quote = (request.query_params.get('quote') or '').strip()

        data = document.extracted_data or {}
        page_map = data.get('_optimized_page_map') if isinstance(data, dict) else None
        optimized_page_num = None

        if document.optimized_file and isinstance(page_map, list):
            try:
                idx = page_map.index(raw_page_num)
                optimized_page_num = idx + 1
            except ValueError:
                optimized_page_num = None

        if document.optimized_file and optimized_page_num is not None:
            pdf_path = document.optimized_file.path
            render_page_num = optimized_page_num
        else:
            pdf_path = document.file.path
            render_page_num = raw_page_num

        try:
            import re as _re

            doc = fitz.open(pdf_path)
            if render_page_num < 1 or render_page_num > len(doc):
                doc.close()
                return Response({'error': 'Page number out of range'}, status=status.HTTP_400_BAD_REQUEST)

            page = doc.load_page(render_page_num - 1)
            pix = page.get_pixmap(dpi=150)
            img_bytes = pix.tobytes('png')
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')

            matched_bboxes = []
            if quote:
                # Strip markdown formatting so the text matches the PDF's raw content
                def _strip_markdown(text):
                    text = _re.sub(r'#{1,6}\s*', '', text)          # headings
                    text = _re.sub(r'\*{1,3}([^*\n]+)\*{1,3}', r'\1', text)  # bold/italic
                    text = _re.sub(r'`[^`]*`', '', text)             # inline code
                    text = _re.sub(r'\|', ' ', text)                 # table pipes
                    text = _re.sub(r'-{3,}', ' ', text)              # HR / table dividers
                    text = _re.sub(r'!\[.*?\]\(.*?\)', '', text)     # images
                    text = _re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # links
                    text = _re.sub(r'\s+', ' ', text).strip()
                    return text

                clean_quote = _strip_markdown(quote)

                # Build a list of increasingly short candidates to try
                candidates = []
                # 1. Full cleaned quote (up to 200 chars to keep search fast)
                if clean_quote:
                    candidates.append(clean_quote[:200])
                # 2. First 100 chars
                if len(clean_quote) > 60:
                    candidates.append(clean_quote[:100])
                # 3. First 12 words
                words = clean_quote.split()
                if len(words) >= 6:
                    candidates.append(' '.join(words[:12]))
                # 4. First 8 words
                if len(words) >= 4:
                    candidates.append(' '.join(words[:8]))
                # 5. Each sentence that is >= 25 chars
                for sent in _re.split(r'(?<=[.!?。])\s+', clean_quote):
                    sent = sent.strip()
                    if 25 <= len(sent) <= 200:
                        candidates.append(sent)
                # 6. Per-line candidates (Mistral OCR produces clean lines per record)
                for line in clean_quote.splitlines():
                    line = line.strip()
                    if 20 <= len(line) <= 200:
                        candidates.append(line)

                # Deduplicate while preserving order
                seen = set()
                deduped = []
                for c in candidates:
                    key = c.lower()
                    if key not in seen:
                        seen.add(key)
                        deduped.append(c)

                rects = []
                for q in deduped:
                    if not q:
                        continue
                    try:
                        found = page.search_for(q, quads=False)
                    except Exception:
                        found = []
                    if found:
                        rects = found
                        break

                pr = page.rect
                pw = max(float(pr.width), 1.0)
                ph = max(float(pr.height), 1.0)
                for r in rects[:10]:
                    matched_bboxes.append([
                        max(0, min(1000, float(r.y0) / ph * 1000)),
                        max(0, min(1000, float(r.x0) / pw * 1000)),
                        max(0, min(1000, float(r.y1) / ph * 1000)),
                        max(0, min(1000, float(r.x1) / pw * 1000)),
                    ])

            doc.close()

            return Response({
                'raw_page_number': raw_page_num,
                'render_page_number': render_page_num,
                'image': f'data:image/png;base64,{img_base64}',
                'width': pix.width,
                'height': pix.height,
                'quote': quote,
                'matched_bboxes': matched_bboxes,
            })

        except Exception as e:
            logger.error(f"Error building page context: {str(e)}")
            return Response(
                {'error': f'Failed to build page context: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


@api_view(['GET'])
def hello_world(request):
    """
    A simple API endpoint that returns a hello message
    """
    serializer = MessageSerializer(data={'message': 'Hello from Django!'})
    serializer.is_valid()
    return Response(serializer.data)


@api_view(['GET'])
def health_check(request):
    """
    Health check endpoint
    """
    return Response({
        'status': 'healthy',
        'service': 'IDP Backend API'
    })