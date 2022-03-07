from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import authenticate
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm
from .forms import UserRegisterForm
from django.core.mail import send_mail
from django.core.mail import EmailMultiAlternatives
from django.template.loader import get_template
from django.template import Context
import os

def index(request):
    return render(request, 'account/home.html', {'title': 'Home'})

def atm_branch(request):
    return render(request, 'account/atm_branch.html', {'title':'ATM Branches'})

def contact_us(request):
    return render(request, 'account/contact_us.html', {'title':'Contact Us'})

def about_us(request):
    return render(request, 'account/about_us.html', {'title':'About Us'})


def user_panel(request):
    # set_save_directory = r'static/'
    # os.chdir(set_save_directory)

    file = open(r'static/number.txt', 'r')
    number = int(file.read())
    file.close()
    number = number - 1

    context = {
        'number': number,
        'title': 'User Panel'
    }
    return render(request, 'account/admin_user.html', context=context)

def register(request):
    if request.method == 'POST':
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            email = form.cleaned_data.get('email')
            ###############################mail system ####################
            # htmly = get_template('account/Email.html')
            # d = {'username': username }
            # subject, from_email, to = 'welcome', 'rafsun.sheikh@gmail.com', email
            # html_content = htmly.render(d)
            # msg = EmailMultiAlternatives(subject, html_content, from_email, [to])
            # msg.attach_alternative(html_content, "text/html")
            # msg.send()
            # ###################################################################
            # messages.success(request, f'Your account has been created ! You are not able to log in')
            messages.success(request, 'Account Created successfully! Welcome {}'.format(username) )
            return redirect('login')

    else:
        form = UserRegisterForm()
    return render(request, 'account/register.html', {'form' : form, 'title' : 'register here'})

def user_login(request):
    if request.method == 'POST':

        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username = username, password = password)
        if user is not None:
            form = login(request, user )
            messages.success(request, f' welcome {username}')
            return redirect('index')
        else:
            messages.info(request, f'account done not exit please sign in')
    form = AuthenticationForm()
    return render(request, 'account/login.html', {'form' : form, 'title': 'log in'})



