from rest_framework import generics
from .models import AnalysisRun
from .serializers import AnalysisRunSerializer

class AnalyzeView(generics.ListCreateAPIView):
    queryset = AnalysisRun.objects.all()
    serializer_class = AnalysisRunSerializer
