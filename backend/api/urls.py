from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

# Create router for ViewSets
router = DefaultRouter()
router.register(r'documents', views.DocumentViewSet, basename='document')

urlpatterns = [
    # API endpoints
    path('', include(router.urls)),
    
    # Legacy/utility endpoints
    path('hello/', views.hello_world, name='hello_world'),
    path('health/', views.health_check, name='health_check'),
]

