from flask import Blueprint, render_template, request, flash, jsonify

auth = Blueprint('auth', __name__)

@auth.route('/login', methods=['GET', 'POST'])
def login():
    return  "<p>Login</p>"

@auth.route('/logout', methods=['GET', 'POST'])
def logout():
    return  "<p>Logout</p>"

@auth.route('/sign-up', methods=['GET', 'POST'])
def sign_up():
    return  "<p>Sign Up</p>"
