from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.home_view, name='home'),
    path('login/', views.login_view, name='login'),
    path('approver/', views.approver_dashboard, name='approver_dashboard'),
    path('substitute/', views.substitute_dashboard, name='substitute_dashboard'),
    path('staff/', views.staff_dashboard, name='staff_dashboard'),
    path('logout/', views.logout_view, name='logout'),
    path('approve/<int:expense_id>/', views.approve_expense, name='approve_expense'),
    path('reject/<int:expense_id>/', views.reject_expense, name='reject_expense'),
    path('edit_expense/<int:expense_id>/', views.edit_expense, name='edit_expense'),
    path('toggle-presence/', views.toggle_presence, name='toggle_presence'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
