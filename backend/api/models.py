from django.db import models
from django.contrib.postgres.fields import ArrayField
import json
from pgvector.django import VectorField, HnswIndex
from django.contrib.postgres.indexes import GinIndex
from django.contrib.postgres.search import SearchVectorField
class DocumentChunk(models.Model):
    document = models.ForeignKey('Document', on_delete=models.CASCADE, related_name='chunks')
    content = models.TextField()
    page_number = models.IntegerField()
    embedding = VectorField(dimensions=1024)
    created_at = models.DateTimeField(auto_now_add=True)
    search_vector = SearchVectorField(null=True)
    class Meta:
        indexes = [
            # HNSW Index for fast approximate nearest neighbor search
            HnswIndex(
                name='chunk_embedding_idx',
                fields=['embedding'],
                m=16,               # Max connections per layer (Default 16)
                ef_construction=64, # Size of dynamic candidate list (Default 64)
                opclasses=['vector_cosine_ops'], # Optimize for Cosine Similarity
            ),
            GinIndex(fields=['search_vector'], name='chunk_search_vector_idx')
        ]

    def __str__(self):
        return f"Chunk {self.id} - Doc {self.document.file_name} (Page {self.page_number})"
class Document(models.Model):
    """
    Model to store document metadata and extracted financial data from PDFs
    """
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]

    OCR_MODEL_CHOICES = [
        ('gemini', 'Gemini 2.0 Flash'),
        ('mistral', 'Mistral Large'),
        ('mistral-ocr', 'Mistral OCR + Small'),
    ]
    
    MODEL_CHOICES = [
        ('gemini', 'Gemini 2.0 Flash'),
        ('mistral', 'Mistral OCR 3')
    ]
    
    # File information
    file = models.FileField(upload_to='documents/%Y/%m/%d/')
    file_name = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed_at = models.DateTimeField(null=True, blank=True)
    
    # Processing status
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    ocr_model = models.CharField(max_length=20, choices=OCR_MODEL_CHOICES, default='gemini')
    error_message = models.TextField(blank=True, null=True)

    # RAG ingestion status (chunking + embedding)
    RAG_STATUS_CHOICES = [
        ('not_started', 'Not Started'),
        ('queued', 'Queued'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    rag_status = models.CharField(max_length=20, choices=RAG_STATUS_CHOICES, default='not_started')
    rag_progress = models.PositiveSmallIntegerField(default=0, help_text='0-100')
    rag_error_message = models.TextField(blank=True, null=True)
    rag_started_at = models.DateTimeField(null=True, blank=True)
    rag_completed_at = models.DateTimeField(null=True, blank=True)
    
    # Extracted data (stored as JSON)
    extracted_data = models.JSONField(null=True, blank=True)

    # Persisted chat history for the document (list of messages)
    # Stored as JSON so the frontend can restore conversations when reopening.
    chat_history = models.JSONField(default=list, blank=True)
    
    # Optimized PDF file (containing only relevant pages)
    optimized_file = models.FileField(upload_to='optimized_documents/%Y/%m/%d/', null=True, blank=True)
    
    # Store the extracted Markdown from OCR
    markdown_file = models.FileField(upload_to='markdown_outputs/%Y/%m/%d/', null=True, blank=True)
    
    # For evaluation purposes
    confidence_score = models.FloatField(null=True, blank=True)
    
    # Edit tracking
    edit_count = models.IntegerField(default=0)
    last_edited_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-uploaded_at']
        indexes = [
            models.Index(fields=['-uploaded_at']),
            models.Index(fields=['status']),
            models.Index(fields=['rag_status']),
        ]
    
    def __str__(self):
        return f"{self.file_name} - {self.status}"
    
    def get_extracted_data(self):
        """Helper method to safely get extracted data"""
        return self.extracted_data or {}
    
    def set_extracted_data(self, data):
        """Helper method to set extracted data"""
        self.extracted_data = data


class ExtractedFundData(models.Model):
    """
    Normalized model to store structured fund data for better querying
    """
    document = models.OneToOneField(Document, on_delete=models.CASCADE, related_name='fund_data')
    
    # Extracted fields - Identity info
    fund_name = models.CharField(max_length=500, blank=True, null=True)
    fund_code = models.CharField(max_length=100, blank=True, null=True)
    fund_type = models.CharField(max_length=200, blank=True, null=True)
    legal_structure = models.CharField(max_length=200, blank=True, null=True)
    license_number = models.CharField(max_length=200, blank=True, null=True)
    regulator = models.CharField(max_length=300, blank=True, null=True)
    management_company = models.CharField(max_length=500, blank=True, null=True)
    custodian_bank = models.CharField(max_length=300, blank=True, null=True)
    fund_supervisor = models.CharField(max_length=300, blank=True, null=True)
    
    # Fee information (increased length to handle longer Vietnamese descriptions)
    management_fee = models.CharField(max_length=500, blank=True, null=True)
    subscription_fee = models.CharField(max_length=500, blank=True, null=True)
    redemption_fee = models.CharField(max_length=500, blank=True, null=True)
    switching_fee = models.CharField(max_length=500, blank=True, null=True)

    # Investment objective & strategy
    investment_objective = models.TextField(blank=True, null=True)
    investment_strategy = models.TextField(blank=True, null=True)
    investment_style = models.CharField(max_length=100, blank=True, null=True)  # active/passive/unknown
    sector_focus = models.TextField(blank=True, null=True)
    benchmark = models.CharField(max_length=300, blank=True, null=True)

    # Additional fee info
    total_expense_ratio = models.CharField(max_length=500, blank=True, null=True)
    custody_fee = models.CharField(max_length=500, blank=True, null=True)
    audit_fee = models.CharField(max_length=500, blank=True, null=True)
    supervisory_fee = models.CharField(max_length=500, blank=True, null=True)
    other_expenses = models.TextField(blank=True, null=True)

    # Risk factors (high-level textual descriptions)
    concentration_risk = models.TextField(blank=True, null=True)
    liquidity_risk = models.TextField(blank=True, null=True)
    interest_rate_risk = models.TextField(blank=True, null=True)

    # Operational details (Thông tin giao dịch)
    trading_frequency = models.CharField(max_length=200, blank=True, null=True)
    cut_off_time = models.CharField(max_length=200, blank=True, null=True)
    nav_calculation_frequency = models.CharField(max_length=200, blank=True, null=True)
    nav_publication = models.CharField(max_length=300, blank=True, null=True)
    settlement_cycle = models.CharField(max_length=200, blank=True, null=True)

    # Valuation & pricing
    valuation_method = models.TextField(blank=True, null=True)
    pricing_source = models.TextField(blank=True, null=True)

    # Investment restrictions & limits
    investment_restrictions = models.TextField(blank=True, null=True)
    borrowing_limit = models.CharField(max_length=200, blank=True, null=True)
    leverage_limit = models.CharField(max_length=200, blank=True, null=True)

    # Investor rights & distribution
    investor_rights = models.TextField(blank=True, null=True)
    distribution_agent = models.CharField(max_length=500, blank=True, null=True)
    sales_channels = models.TextField(blank=True, null=True)

    # Governance / partners
    auditor = models.CharField(max_length=300, blank=True, null=True)
    
    # Portfolio and table data (stored as JSON arrays)
    portfolio = models.JSONField(default=list, blank=True)
    nav_history = models.JSONField(default=list, blank=True)
    dividend_history = models.JSONField(default=list, blank=True)

    # Extended structured data (JSON)
    asset_allocation = models.JSONField(default=dict, blank=True)
    minimum_investment = models.JSONField(default=dict, blank=True)
    
    # Bounding boxes for extracted fields (for PDF highlighting)
    bboxes = models.JSONField(default=dict, blank=True)
    
    analysis_report = models.JSONField(null=True, blank=True, help_text="AI-generated investment evaluation")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"Fund Data for {self.document.file_name}"
    
    class Meta:
        verbose_name = "Extracted Fund Data"
        verbose_name_plural = "Extracted Fund Data"


class DocumentChangeLog(models.Model):
    """
    Model to track all changes made to a document's extracted data
    """
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name='change_logs')
    
    # Change metadata
    changed_at = models.DateTimeField(auto_now_add=True)
    user_comment = models.TextField(blank=True, null=True)
    
    # Change details (stored as JSON)
    changes = models.JSONField(default=dict)  # {field_path: {old: value, new: value}}
    
    class Meta:
        ordering = ['-changed_at']
        indexes = [
            models.Index(fields=['document', '-changed_at']),
        ]
    
    def __str__(self):
        return f"Change for {self.document.file_name} at {self.changed_at}"
