from django.db import models

class AnalysisRun(models.Model):
    run_name = models.CharField(max_length=100, unique=True)
    video_url = models.URLField()
    analyzed_video_url = models.URLField(blank=True, null=True) 
    timestamp = models.DateTimeField(auto_now_add=True)

    flocking_index = models.CharField(max_length=50)
    avg_tel = models.CharField(max_length=50)
    avg_ta = models.CharField(max_length=50)
    avg_group_speed = models.CharField(max_length=50)

    def __str__(self):
        return self.run_name
