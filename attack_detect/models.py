from django.db import models


class SuspiciousUser(models.Model):
    user_id = models.IntegerField()
    objects = models.Manager()
    is_return = models.BooleanField(default=False)


class Rate(models.Model):
    user_id = models.IntegerField()
    movie_id = models.IntegerField()
    objects = models.Manager()
    RATING = (
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 4),
        (5, 5),
    )
    rating = models.IntegerField(choices=RATING)
    timestamp = models.IntegerField()
