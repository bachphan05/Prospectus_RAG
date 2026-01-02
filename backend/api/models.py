from django.db import models
from django.contrib.postgres.fields import ArrayField
import json


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
    
    # Extracted data (stored as JSON)
    extracted_data = models.JSONField(null=True, blank=True)
    
    # Optimized PDF file (containing only relevant pages)
    optimized_file = models.FileField(upload_to='optimized_documents/%Y/%m/%d/', null=True, blank=True)
    
    # For evaluation purposes
    confidence_score = models.FloatField(null=True, blank=True)
    
    class Meta:
        ordering = ['-uploaded_at']
        indexes = [
            models.Index(fields=['-uploaded_at']),
            models.Index(fields=['status']),
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
    management_company = models.CharField(max_length=500, blank=True, null=True)
    custodian_bank = models.CharField(max_length=300, blank=True, null=True)
    
    # Fee information
    management_fee = models.CharField(max_length=200, blank=True, null=True)
    subscription_fee = models.CharField(max_length=200, blank=True, null=True)
    redemption_fee = models.CharField(max_length=200, blank=True, null=True)
    switching_fee = models.CharField(max_length=200, blank=True, null=True)
    
    # Portfolio and table data (stored as JSON arrays)
    portfolio = models.JSONField(default=list, blank=True)
    nav_history = models.JSONField(default=list, blank=True)
    dividend_history = models.JSONField(default=list, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"Fund Data for {self.document.file_name}"
    
    class Meta:
        verbose_name = "Extracted Fund Data"
        verbose_name_plural = "Extracted Fund Data"
