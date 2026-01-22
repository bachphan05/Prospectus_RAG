from rest_framework import serializers
from .models import Document, ExtractedFundData, DocumentChangeLog


class ExtractedFundDataSerializer(serializers.ModelSerializer):
    """Serializer for structured fund data"""
    
    class Meta:
        model = ExtractedFundData
        fields = [
            'fund_name', 'fund_code',
            'fund_type', 'legal_structure', 'license_number', 'regulator',
            'management_company', 'custodian_bank', 'fund_supervisor',
            'management_fee', 'subscription_fee', 'redemption_fee', 'switching_fee',
            'total_expense_ratio',
            'custody_fee', 'audit_fee', 'supervisory_fee', 'other_expenses',
            'investment_objective', 'investment_strategy', 'investment_style', 'sector_focus', 'benchmark',
            'investment_restrictions', 'borrowing_limit', 'leverage_limit',
            'concentration_risk', 'liquidity_risk', 'interest_rate_risk',
            'trading_frequency', 'cut_off_time', 'nav_calculation_frequency', 'nav_publication', 'settlement_cycle',
            'valuation_method', 'pricing_source',
            'investor_rights', 'distribution_agent', 'sales_channels',
            'auditor',
            'asset_allocation', 'minimum_investment',
            'portfolio', 'nav_history', 'dividend_history',
            'created_at', 'updated_at'
        ]
        read_only_fields = ['created_at', 'updated_at']


class DocumentChangeLogSerializer(serializers.ModelSerializer):
    """Serializer for document change logs"""
    
    class Meta:
        model = DocumentChangeLog
        fields = ['id', 'document', 'changed_at', 'user_comment', 'changes']
        read_only_fields = ['id', 'document', 'changed_at']


class ChatRequestSerializer(serializers.Serializer):
    """Serializer for RAG chat requests"""
    query = serializers.CharField(required=True, max_length=1000)
    history = serializers.ListField(required=False, default=list, allow_empty=True)


class ChatResponseSerializer(serializers.Serializer):
    """Serializer for RAG chat responses"""
    answer = serializers.CharField()
    query = serializers.CharField()
    chunks_count = serializers.IntegerField(required=False)


class ChatHistorySerializer(serializers.Serializer):
    """Serializer for persisting chat history per document"""
    history = serializers.ListField(required=True, allow_empty=True)


class DocumentSerializer(serializers.ModelSerializer):
    """Serializer for document metadata and extracted data"""
    fund_data = ExtractedFundDataSerializer(read_only=True)
    file_url = serializers.SerializerMethodField()
    optimized_file_url = serializers.SerializerMethodField()
    
    class Meta:
        model = Document
        fields = [
            'id', 
            'file', 
            'file_name', 
            'uploaded_at', 
            'processed_at',
            'status',
            'ocr_model',
            'error_message',
            'extracted_data',
            'confidence_score',
            'fund_data',
            'file_url',
            'optimized_file_url',
            'edit_count',
            'last_edited_at'
        ]
        read_only_fields = ['uploaded_at', 'processed_at', 'status', 'confidence_score', 'fund_data', 'edit_count', 'last_edited_at']
    
    def get_file_url(self, obj):
        """Get the full URL for the uploaded file"""
        request = self.context.get('request')
        if obj.file and hasattr(obj.file, 'url'):
            if request is not None:
                return request.build_absolute_uri(obj.file.url)
            return obj.file.url
        return None

    def get_optimized_file_url(self, obj):
        """Get the full URL for the optimized file"""
        request = self.context.get('request')
        if obj.optimized_file and hasattr(obj.optimized_file, 'url'):
            if request is not None:
                return request.build_absolute_uri(obj.optimized_file.url)
            return obj.optimized_file.url
        return None


class DocumentUploadSerializer(serializers.ModelSerializer):
    """Serializer for uploading documents"""
    
    class Meta:
        model = Document
        fields = ['file', 'file_name', 'ocr_model']
    
    def validate_file(self, value):
        """Validate uploaded file"""
        # Check file extension
        if not value.name.lower().endswith('.pdf'):
            raise serializers.ValidationError("Only PDF files are allowed.")
        
        # Check file size (max 100MB)
        max_size = 100 * 1024 * 1024  # 100MB
        if value.size > max_size:
            raise serializers.ValidationError(f"File size cannot exceed 100MB. Current size: {value.size / (1024*1024):.2f}MB")
        
        return value
    
    def create(self, validated_data):
        """Create document instance with pending status"""
        if not validated_data.get('file_name'):
            validated_data['file_name'] = validated_data['file'].name
        
        # Default to gemini if not specified
        if 'ocr_model' not in validated_data:
            validated_data['ocr_model'] = 'gemini'
        
        validated_data['status'] = 'pending'
        return super().create(validated_data)


class DocumentListSerializer(serializers.ModelSerializer):
    """Lightweight serializer for document list view"""
    fund_name = serializers.SerializerMethodField()
    fund_code = serializers.SerializerMethodField()
    
    class Meta:
        model = Document
        fields = ['id', 'file_name', 'uploaded_at', 'status', 'fund_name', 'fund_code', 'edit_count', 'last_edited_at']
        read_only_fields = ['id', 'file_name', 'uploaded_at', 'status', 'edit_count', 'last_edited_at']
    
    def get_fund_name(self, obj):
        """Get fund name from related fund_data or extracted_data"""
        if hasattr(obj, 'fund_data') and obj.fund_data:
            return obj.fund_data.fund_name
        return obj.extracted_data.get('fund_name') if obj.extracted_data else None
    
    def get_fund_code(self, obj):
        """Get fund code from related fund_data or extracted_data"""
        if hasattr(obj, 'fund_data') and obj.fund_data:
            return obj.fund_data.fund_code
        return obj.extracted_data.get('fund_code') if obj.extracted_data else None


class MessageSerializer(serializers.Serializer):
    message = serializers.CharField()
