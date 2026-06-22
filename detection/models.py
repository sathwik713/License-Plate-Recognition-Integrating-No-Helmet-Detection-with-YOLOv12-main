from django.db import models

class Detection(models.Model):
    date = models.DateField(auto_now_add=True)
    time = models.TimeField(auto_now_add=True)
    helmet_status = models.CharField(max_length=20, choices=[('Helmet', 'Helmet'), ('No Helmet', 'No Helmet')])
    license_plate_text = models.CharField(max_length=20, blank=True, null=True)
    image_path = models.ImageField(upload_to='frames/', blank=True, null=True)
    video_source = models.CharField(max_length=255)
    fine_amount = models.DecimalField(max_digits=10, decimal_places=2, default=0)
    challan_pdf = models.FileField(upload_to='challans/', blank=True, null=True)

    def __str__(self):
        return f"{self.helmet_status} - {self.license_plate_text} at {self.time}"