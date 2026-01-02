from rest_framework import serializers
from .models import Document, ExtractedFundData


class ExtractedFundDataSerializer(serializers.ModelSerializer):
    """Serializer for structured fund data"""
    
    class Meta:
        model = ExtractedFundData
        fields = [
            'fund_name', 'fund_code', 'management_company', 'custodian_bank',
            'management_fee', 'subscription_fee', 'redemption_fee', 'switching_fee',
            'portfolio', 'nav_history', 'dividend_history',
            'created_at', 'updated_at'
        ]
        read_only_fields = ['created_at', 'updated_at']


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
            'optimized_file_url'
        ]
        read_only_fields = ['uploaded_at', 'processed_at', 'status', 'extracted_data', 'confidence_score', 'fund_data']
    
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
        fields = ['id', 'file_name', 'uploaded_at', 'status', 'fund_name', 'fund_code']
        read_only_fields = ['id', 'file_name', 'uploaded_at', 'status']
    
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
