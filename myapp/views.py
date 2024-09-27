# myapp/views.py
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from .forms import ExpenseForm
from .models import ApproverSubstitute, Expense, Approver, ApproverCategory,Category,Substitute
from django.http import JsonResponse
import json
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt



def login_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            if user.userprofile.role == 'approver':
                return redirect('approver_dashboard')
            elif user.userprofile.role == 'staff':
                return redirect('staff_dashboard')
            else:
                return redirect('substitute_dashboard')
        else:
            return render(request, 'myapp/login.html', {'error': 'Invalid username or password'})
    else:
        return render(request, 'myapp/login.html')

@login_required
def approver_dashboard(request):
    approver = get_object_or_404(Approver, user=request.user)
    
    # Get the categories the approver is responsible for
    approver_categories = ApproverCategory.objects.filter(approver=approver)
    categories = [ac.category.name for ac in approver_categories]
    
    # Get the expenses based on the categories
    expenses = Expense.objects.filter(category__in=categories)

    return render(request, 'myapp/approver_dashboard.html', {
        'logged_in_user': request.user.username,
        'expenses': expenses
    })


@login_required
def staff_dashboard(request):
    if request.method == 'POST':
        form = ExpenseForm(request.POST, request.FILES)
        if form.is_valid():
            expense = form.save(commit=False)
            expense.user = request.user
            expense.description = f"{expense.category} expense on {expense.date} by {request.user.username}"

            # Automatic approval logic
            if expense.category in ['Wifi', 'Meals'] and expense.amount < 200:
                expense.status = 'approved'
            else:
                expense.status = 'pending'  # Default to pending if not automatically approved

            expense.save()

            # Check the approver's presence and handle the substitute
            handle_expense_approval(expense)

            return redirect('staff_dashboard')
    else:
        form = ExpenseForm()

    expenses = Expense.objects.filter(user=request.user)
    return render(request, 'myapp/staff_dashboard.html', {
        'form': form,
        'expenses': expenses,
        'logged_in_user': request.user.username
    })

def handle_expense_approval(expense):
    # Get the approver for the expense category
    approver = Approver.objects.filter(approvercategory__category__name=expense.category).first()

    if approver and not approver.is_present:
        # Find the substitute for the approver
        approver_substitute = ApproverSubstitute.objects.filter(approver=approver).first()
        if approver_substitute:
            # Display the expenses to the substitute
            expense.substitute = approver_substitute.substitute
            expense.save()
            
def home_view(request):
    return redirect('login')

def logout_view(request):
    logout(request)
    return redirect('login')

@login_required
def approve_expense(request, expense_id):
    expense = Expense.objects.get(id=expense_id)
    expense.status = 'approved'
    expense.save()
    return redirect('approver_dashboard')

@login_required
def reject_expense(request, expense_id):
    expense = Expense.objects.get(id=expense_id)
    expense.status = 'rejected'
    expense.save()
    return redirect('approver_dashboard')

def update_expense_status(request):
    if request.method == 'POST':
        data = json.loads(request.body)
        expense_id = data.get('id')
        status = data.get('status')
        try:
            expense = Expense.objects.get(id=expense_id)
            expense.status = status
            expense.save()
            return JsonResponse({'success': True})
        except Expense.DoesNotExist:
            return JsonResponse({'success': False})
    return JsonResponse({'success': False})

@login_required
def edit_expense(request, expense_id):
    expense = get_object_or_404(Expense, id=expense_id)
    if request.method == 'POST':
        form = ExpenseForm(request.POST, request.FILES, instance=expense)
        if form.is_valid():
            category = form.cleaned_data['category']
            amount = form.cleaned_data['amount']
            date = form.cleaned_data['date']
            description = f"{category} expense of {amount} on {date}"
            form.instance.description = description
            
            # Save the updated expense
            form.save()
            return redirect('staff_dashboard')
    else:
        form = ExpenseForm(instance=expense)

    return render(request, 'myapp/edit_expense.html', {'form': form})


@csrf_exempt
@login_required
def toggle_presence(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            is_present = data.get('is_present', False)
            user = request.user
            try:
                approver = Approver.objects.get(user=user)
                approver.is_present = is_present
                approver.save()
                return JsonResponse({'success': True})
            except Approver.DoesNotExist:
                return JsonResponse({'success': False, 'error': 'Approver does not exist'})
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    return JsonResponse({'success': False, 'error': 'Invalid request'})

@login_required
def substitute_dashboard(request):
    user = request.user
    substitute = get_object_or_404(Substitute, user=user)

    # Get the approvers for whom this user is a substitute
    approver_substitutes = ApproverSubstitute.objects.filter(substitute=substitute)
    approvers = [asub.approver for asub in approver_substitutes]

    # Get the categories these approvers are responsible for
    approver_categories = ApproverCategory.objects.filter(approver__in=approvers)
    categories = [ac.category.name for ac in approver_categories]

    # Get the approvers who are absent
    absent_approvers = Approver.objects.filter(id__in=[a.id for a in approvers], is_present=False)

    # If any approvers are absent, display expenses in their categories
    if absent_approvers.exists():
        absent_approver_ids = absent_approvers.values_list('id', flat=True)
        expenses = Expense.objects.filter(category__in=categories, user__userprofile__role='staff')
    else:
        expenses = Expense.objects.none()  # No expenses to show if all approvers are present

    return render(request, 'myapp/substitute_dashboard.html', {
        'logged_in_user': request.user.username,
        'expenses': expenses
    })
