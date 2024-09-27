# myapp/forms.py
from django import forms
from .models import Expense,Approver

class ExpenseForm(forms.ModelForm):
    class Meta:
        model = Expense
        fields = ['date', 'amount', 'category', 'image']
        widgets = {
            'date': forms.DateInput(attrs={'type': 'date', 'class': 'debug'}),
            'amount': forms.NumberInput(attrs={'step': '0.01', 'class': 'debug'}),
            'category': forms.Select(choices=Expense.CATEGORY_CHOICES, attrs={'class': 'debug'}),
            'image': forms.ClearableFileInput(attrs={'class': 'debug'}),
        }

class ApproverPresenceForm(forms.ModelForm):
    class Meta:
        model = Approver
        fields = ['is_present']
        widgets = {
            'is_present': forms.CheckboxInput(attrs={'class': 'presence-toggle'}),
        }