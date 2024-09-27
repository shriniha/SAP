# myapp/admin.py
from django.contrib import admin
from .models import UserProfile, Expense
from .models import Approver, Category, ApproverCategory,Substitute, ApproverSubstitute

@admin.register(Approver)
class ApproverAdmin(admin.ModelAdmin):
    list_display = ['name', 'email', 'user','is_present']
    search_fields = ['name', 'email', 'user__username']
@admin.register(Category)
class CategoryAdmin(admin.ModelAdmin):
    list_display = ('name',)
    search_fields = ('name',)

@admin.register(ApproverCategory)
class ApproverCategoryAdmin(admin.ModelAdmin):
    list_display = ('approver', 'category')
    search_fields = ('approver__name', 'category__name')
    list_filter = ('approver', 'category')
    list_filter = ('approver', 'category')
    
@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'role','is_present')
    list_filter = ('role',)

# Admin interface for Substitute model
@admin.register(Substitute)
class SubstituteAdmin(admin.ModelAdmin):
    list_display = ('name', 'email')
    search_fields = ('name','email')


# Admin interface for ApproverSubstitute model
@admin.register(ApproverSubstitute)
class ApproverSubstituteAdmin(admin.ModelAdmin):
    list_display = ('approver', 'substitute')
    search_fields = ('approver__name', 'substitute__name')
    list_filter = ('approver', 'substitute')
    list_filter = ('approver', 'substitute')

class ExpenseAdmin(admin.ModelAdmin):
    list_display = ('date', 'amount', 'category', 'description', 'user', 'status')  # Added status to display
    readonly_fields = ('user',)  # Make user read-only

    def get_readonly_fields(self, request, obj=None):
        if obj:  # This is the case when obj is being edited
            return self.readonly_fields + ('user',)
        return self.readonly_fields
    
admin.site.register(Expense, ExpenseAdmin)

