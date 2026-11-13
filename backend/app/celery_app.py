from celery import Celery

from .config import get_settings

settings = get_settings()

celery_app = Celery("docrag", broker=settings.redis_url, backend=settings.redis_url)
celery_app.conf.update(
    task_acks_late=True,  # re-deliver if a worker dies mid-task
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    broker_transport_options={"visibility_timeout": 3600},
    result_expires=3600,
    task_time_limit=600,
    imports=("app.tasks",),
)
