from flask import Flask, request, jsonify, render_template,redirect, url_for, make_response
import cv2
import numpy as np
import mediapipe as mp
from flask_jwt_extended import JWTManager, create_access_token,jwt_required, set_access_cookies, unset_jwt_cookies
import threading
import random
import couchdb 
from datetime import timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


app = Flask(__name__)
from flask_cors import CORS
CORS(app, resources={r"/*": {"origins": "*"}})
app.config["JWT_SECRET_KEY"] = "your-secret-key-change-it"  # Change this in production!
app.config["JWT_TOKEN_LOCATION"] = ["cookies"]
app.config["JWT_COOKIE_SECURE"] = False  # Set to True in production with HTTPS
app.config["JWT_COOKIE_CSRF_PROTECT"] = False  # Enable CSRF protection in production
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = timedelta(hours=1)  # Token expires after 1 hour
jwt = JWTManager(app)

# Blacklist for revoked tokens (in production, use a database)
revoked_tokens = set()

# Add this to check if token is revoked
@jwt.token_in_blocklist_loader
def check_if_token_revoked(jwt_header, jwt_payload):
    jti = jwt_payload["jti"]
    return jti in revoked_tokens

# Global variable to store the selected condition
selected_condition = None

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
MIn_hand_size = 0.3
Max_hand_size = 0.6

def is_hand_in_range(hand_landmarks):
    x_values = []
    y_values = []
    for lm in hand_landmarks.landmark:
        x_values.append(lm.x)
        y_values.append(lm.y)
        hand_width = max(x_values) - min(x_values)
        hand_height = max(y_values) - min(y_values)
        hand_size = max(hand_width, hand_height)
    return MIn_hand_size <= hand_size <= Max_hand_size

def is_hand_straight(hand_landmarks):
    def get_angle(a, b, c):
        ba = np.array([a.x - b.x, a.y - b.y])
        bc = np.array([c.x - b.x, c.y - b.y])
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc)) 
        return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

    fingers = [
        [mp_hands.HandLandmark.INDEX_FINGER_MCP, mp_hands.HandLandmark.INDEX_FINGER_PIP, mp_hands.HandLandmark.INDEX_FINGER_TIP],
        [mp_hands.HandLandmark.MIDDLE_FINGER_MCP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
        [mp_hands.HandLandmark.RING_FINGER_MCP, mp_hands.HandLandmark.RING_FINGER_PIP, mp_hands.HandLandmark.RING_FINGER_TIP],
        [mp_hands.HandLandmark.PINKY_MCP, mp_hands.HandLandmark.PINKY_PIP, mp_hands.HandLandmark.PINKY_TIP]
    ]

    angles = []
    for mcp, pip, tip in fingers:
        mcp_landmark = hand_landmarks.landmark[mcp]
        pip_landmark = hand_landmarks.landmark[pip]
        tip_landmark = hand_landmarks.landmark[tip]

        # Angle between MCP, PIP, and TIP (should be near 180° for a straight hand)
        angle = get_angle(mcp_landmark, pip_landmark, tip_landmark)
        angles.append(angle)
    # Check if all angles are near 170-180 degrees (fully extended)
    return all(170 <= angle <= 180 for angle in angles)

def gesture_recognition(frame, hand_landmarks, pbutton):
    h, w, _ = frame.shape
    if hand_landmarks and pbutton:  # Only process if we have hand landmarks and a condition
        if pbutton == 'Eye nerve':
            # Eyes on index
            index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
            index_x, index_y = int((index_pip.x) * w), int((index_pip.y) * h + 25)
            cv2.circle(frame, (index_x, index_y), 5, (255, 0, 0), -1)
            cv2.putText(frame, "Eyes", (index_x + 50, index_y - 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Eyes and ears between middle and ring
            middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
            mid_x, mid_y = int((middle_mcp.x) * w), int((middle_mcp.y) * h + 25)
            cv2.circle(frame, (mid_x, mid_y), 5, (255, 0, 0), -1)
        
        elif pbutton == 'Ear nerve':
            # Ears on pinky finger
            pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP]
            pink_x, pink_y = int((pinky_mcp.x) * w), int((pinky_mcp.y) * h + 17)
            cv2.circle(frame, (pink_x, pink_y), 5, (255, 0, 0), -1)

            # Ears on ring finger
            ring_pip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
            ring_x, ring_y = int((ring_pip.x) * w - 3), int((ring_pip.y) * h + 15)
            cv2.circle(frame, (ring_x, ring_y), 5, (255, 0, 0), -1)
        
        elif pbutton == 'Sinuses':
            pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            mid_x, mid_y = int((pinky_mcp.x) * w + 2), int((pinky_mcp.y) * h - 10)
            cv2.circle(frame, (mid_x, mid_y), 8, (0, 255, 0), -1)
            cv2.putText(frame, "sinuses", (mid_x + 40, mid_y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            ringsinuses = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            x,y=int((ringsinuses.x)*w+2),int((ringsinuses.y)*h-10)
            cv2.circle(frame, (x,y), 8, (0, 255, 0), -1)

            middlesinuses = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            middlex,middley=int((middlesinuses.x)*w+2),int((middlesinuses.y)*h-10)
            cv2.circle(frame, (middlex,middley), 8, (0, 255, 0), -1)

            indexsinuses = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            indexx,indexy=int((indexsinuses.x)*w+2),int((indexsinuses.y)*h-10)
            cv2.circle(frame, (indexx,indexy), 8, (0, 255, 0), -1)

        elif pbutton=='Headache':
            #2 headache
            headache=hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            headachex,headachey=int ((headache.x)*w+4),int ((headache.y)*h-10)
            cv2.circle(frame, (headachex,headachey),8,(250,255,0), -1)

        elif pbutton=='Stress':
            #stress
            stress=hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
            stressx,stressy=int ((stress.x)*w), int((stress.y)*h)
            stressx,stressy=stressx -15,stressy-20
            cv2.circle(frame, (stressx,stressy),10, (200,255,0), -1)

        elif pbutton=='Sleeping Point':
            #sleeping point
            sleep=hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
            sleepx,sleepy=int((sleep.x)*w+15),int ((sleep.y)*h+5)
            cv2.circle(frame,(sleepx,sleepy),6,(150,255,0),-1)

        # common cold
        elif pbutton=='Common Cold':
            cold1=hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP]
            cold2=hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            coldx,coldy=int((cold1.x+cold2.x)/2*w),int((cold1.y+cold2.y)/2*h)
            coldx=coldx+5
            cv2.circle(frame,(coldx,coldy),8,(110,255,0),-1)

        #tension
        elif pbutton=='Tension':
            tension1=hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            tension2=hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]
            tensionx,tensiony=int((tension1.x + tension2.x)/2*w-6),int((tension1.y+tension2.y)/2*h)
            cv2.circle(frame,(tensionx,tensiony),8,(0,0,0),-1)

        #energy
        elif pbutton=='Energy Point':
            energy1=hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            energy2=hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP]
            energyx,energyy=int((energy1.x+energy2.x)/2*w),int((energy1.y+energy2.y)/2*h)
            energyx,energyy=energyx+3,energyy
            cv2.circle(frame,(energyx,energyy),5,(10,255,0),-1)

        #iq level point
        elif pbutton=='IQ Level Point':
            iq=hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]
            iqx,iqy=int((iq.x)*w-8),int((iq.y)*h-8)
            cv2.circle(frame,(iqx,iqy),5,(0,110,10),-1)

        #12 eye problem
        elif pbutton=='Eye Problem':
            eye=hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
            eyex,eyey=int((eye.x)*w+12),int(eye.y*h-8)
            cv2.circle(frame,(eyex,eyey),10,(0,110,10),-1)

        #13 ear problem
        elif pbutton=='Ear Disease':
            ear=hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
            earx,eary=int(ear.x*w),int(ear.y*h)
            cv2.circle(frame,(earx,eary),10,(0,5,100),-1)

        #14 shoulder pain
        elif pbutton=='Shoulder Pain':
            shu=hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
            shux,shuy=int((shu.x)*w+10),int((shu.y)*h+5)
            cv2.circle(frame,(shux,shuy),5,(0,155,255),-1)

        #15 lungs problem
        elif pbutton=='Lungs Problem':
            lung1=hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            lung2=hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
            lungx,lungy=int((lung1.x+lung2.x)/2*w),int((lung1.y+lung2.y)/2*h+10)
            cv2.circle(frame,(lungx,lungy),5,(155,124,125),-1)

        # # 16 trouble
        elif pbutton=='Navel':
            troub1=hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            troub2=hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
            troub3=hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
            troub4=hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
            troubx,trouby=int((troub1.x+troub2.x+troub3.x+troub4.x)/4*w),int((troub1.y+troub2.y+troub3.y+troub4.y)/4*h+20)
            cv2.circle(frame,(troubx,trouby),5,(12,12,255),-1)

        # 17 stomach disease
        elif pbutton=='Stomach Disease':
            stomach1=hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
            stomach2=hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
            stomachx,stomachy=int((stomach1.x+stomach2.x)/2*w+19),int((stomach1.y+stomach2.y)/2*h+10)
            cv2.circle(frame,(stomachx,stomachy),5,(12,12,255),-1)

        # 18 liver disease
        elif pbutton=='Liver Disease':
            liver1=hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
            liver2=hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
            liverx,livery=int((liver1.x+liver2.x)/2*w),int((liver1.y+liver2.y)/2*h+40)
            cv2.circle(frame,(liverx,livery),5,(0,0,15),-1)

        #21 diabities
        elif pbutton=='Diabetes':
            diab1=hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
            diab2=hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            diabx,diaby=int((diab1.x+diab2.x)/2*w+4),int((diab1.y+diab2.y)/2*h+42)
            cv2.circle(frame,(diabx,diaby),5,(15,120,15),-1)

        #22 kidney disease
        elif pbutton=='Kidney Disease':
            kidney1=hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            kidney2=hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
            kidneyx,kidneyy=int((kidney1.x+kidney2.x)/2*w+18),int((kidney1.y+kidney2.y)/2*h+29)
            cv2.circle(frame,(kidneyx,kidneyy),5,(0,50,200),-1)

        #23 Arthritis
        elif pbutton=='Arthritis':
            arth1=hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            arth2=hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            arthx,arthy=int((arth1.x+arth2.x)/2*w-15),int((arth1.y+arth2.y)/2*h+15)
            cv2.circle(frame,(arthx,arthy),5,(50,50,5),-1)

        #24 gall bladeer
        elif pbutton=='Gall Bladder':
            gall1=hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
            gall2=hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
            gallx,gally=int((gall1.x+gall2.x)/2*w),int((gall1.y+gall2.y)/2*h+55)
            cv2.circle(frame,(gallx,gally),5,(0,0,15),-1)

        #27 digestive system disease
        elif pbutton=='Digestive System Disease':
            digestive1=hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
            digestive2=hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            digestivex,digestivey=int((digestive1.x+digestive2.x)/2*w+6),int((digestive1.y+digestive2.y)/2*h+75)
            cv2.circle(frame,(digestivex,digestivey),5,(15,120,255),-1)

        # 28 large intestine disease
        elif pbutton=='large intestine disease':
            large1=hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            large2=hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            largex,largey=int((large1.x+large2.x)/2*w-20),int((large1.y+large2.y)/2*h+28)
            cv2.circle(frame,(largex,largey),5,(5,255,5),-1)

        # 29 increased appetite point
        elif pbutton=='Increased appetite point':
            increase1=hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            increase2=hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            increasex,increasey=int((increase1.x+increase2.x)/2*w-15),int((increase1.y+increase2.y)/2*h+40)
            cv2.circle(frame,(increasex,increasey),5,(255,255,255),-1)

        # 30 interstitial disease point
        elif pbutton=='Interstitial disease point':
            interstitial1=hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            interstitial2=hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            interstitialx,interstitialy=int((interstitial1.x+interstitial2.x)/2*w-12),int((interstitial1.y+interstitial2.y)/2*h+52)
            cv2.circle(frame,(interstitialx,interstitialy),5,(255,255,0),-1)

        # #31 indigestion point
        elif pbutton=='Indigestion point':
            digestive1=hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
            digestive2=hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            digestivex,digestivey=int((digestive1.x+digestive2.x)/2*w+4),int((digestive1.y+digestive2.y)/2*h+105)
            cv2.circle(frame,(digestivex,digestivey),5,(255,120,15),-1)

        # #32 urinary tract disease
        elif pbutton=='Urinary tract disease':
            urinary1=hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
            urinary2=hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            urinaryx,urinaryy=int((urinary1.x+urinary2.x)/2*w+2),int((urinary1.y+urinary2.y)/2*h+125)
            cv2.circle(frame,(urinaryx,urinaryy),5,(255,120,15),-1)

        # #34 appendix
        elif pbutton=='Appendix':
            appendix1=hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
            appendix2=hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
            appendixx,appendixy=int((appendix1.x+appendix2.x)/2*w),int((appendix1.y+appendix2.y)/2*h+137)
            cv2.circle(frame,(appendixx,appendixy),5,(255,0,255),-1)

        #35 hemorrhoid bavasir
        elif pbutton=='Hemorrhoid bavasir':
            hemo1=hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            hemo2=hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC]
            hemox,hemoy=int((hemo1.x+hemo2.x)/2*w+2),int((hemo1.y+hemo2.y)/2*h+35)
            cv2.circle(frame,(hemox,hemoy),5,(0,0,0),-1)




# CouchDB configuration
COUCH_URL = "http://admin:admin123@localhost:5984"
COUCH_DB_NAME = "users"
def generate_otp(length=6):
    global current_otp
    current_otp = ''.join([str(random.randint(0, 9)) for _ in range(length)])
    print(f"Generated OTP: {current_otp}")
    threading.Timer(120, generate_otp).start()  

# Add this function to send an email with OTP (placeholder)
def send_email_otp(email, otp):
    """
    Send OTP to user's email (placeholder function)
    In production, integrate with a real email service like SendGrid, AWS SES, etc.
    """
    print(f"Sending OTP {otp} to {email}")
    # In production, implement actual email sending logic
    return True

@app.route('/verify-otp')
def verify_otp():
    email = request.args.get('email', '')
    if not email:
        return redirect(url_for('forget'))
    return render_template('otp.html', email=email)

# Add this API endpoint for forgot password
# import couchdb

# Connect to CouchDB
couch = couchdb.Server(COUCH_URL)
user_db = couch[COUCH_DB_NAME]
def send_email(to_email, subject, body):
    # SMTP Server settings (example for Gmail)
            smtp_server = 'smtp.gmail.com'
            smtp_port = 587
            smtp_user = 'ashubansal280@gmail.com'  # Replace with your email
            smtp_password = 'yjhxhlxenptlkzdo'  # Replace with your email password or an app-specific password

            # Prepare the email message
            msg = MIMEMultipart()
            msg['From'] = smtp_user
            msg['To'] = to_email
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))

            try:
                # Establish a connection to the SMTP server
                with smtplib.SMTP(smtp_server, smtp_port) as server:
                    server.starttls()  # Secure the connection
                    server.login(smtp_user, smtp_password)
                    server.sendmail(smtp_user, to_email, msg.as_string())  # Send the email
                    return True
            except Exception as e:
                print(f"Error sending email: {e}")
                return False
        

@app.route('/api/forgot-password', methods=['POST'])
def forgot_password_api():
    data = request.json
    email = data.get('email')
    
    if not email:
        return jsonify({"message": "Email is required"}), 400
    
    # Check if email exists in CouchDB
    # CouchDB uses views for querying
    try:
        # Assuming you have a view set up for finding users by email
        results = user_db.view('users/userdetails', key=email)
        
        user_exists = len(results) > 0
        if not user_exists:
            return jsonify({"message": "Email not found in our records"}), 404
            
        # Generate OTP
        global otp;
        generate_otp();
        otp=current_otp;
        
    
        
        # Send email with OTP
        subject = "Password Reset OTP"
        body = f"""
        Hello,
        
        You requested to reset your password.
        Your OTP is: {otp}
        
        This OTP will expire in 1 minutes.
        
        If you did not request this, please ignore this email.
        
        Best regards,
        Your Application Team
        """
                
        email_sent = send_email(email, subject, body)
        if email_sent:
                return jsonify({"message": "OTP sent successfully"}), 200
        else:
                return jsonify({"message": "Failed to send OTP. Please try again later."}), 500
    except Exception as e:
        print(f"Database error: {e}")
    return jsonify({"message": "An error occurred. Please try again."}), 500



# Add this API endpoint to verify OTP
@app.route("/api/verify-otp", methods=['POST'])
def api_verify_otp():
    try:
        data = request.get_json()
        email = data.get('email')
        user_otp = data.get('otp')

        if not email or not user_otp:
            return jsonify({"message": "Email and OTP are required"}), 400
        
        
        if user_otp == current_otp:
            # Success
            return jsonify({"message": "OTP verified successfully"}), 200
        elif user_otp == otp:
            # Success
            return jsonify({"message": "otp expired"}), 401
        else:
            return jsonify({"message": "Invalid OTP"}), 401

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"message": "An unexpected error occurred"}), 500

# Add this API endpoint to reset password
@app.route('/api/reset-password', methods=['POST'])
def api_reset_password():
    try:
        data = request.get_json()
        email = data.get('email')
        reset_token = data.get('reset_token')
        new_password = data.get('new_password')
        
        if not email or not reset_token or not new_password:
            return jsonify({"message": "All fields are required"}), 400


        try:
            import requests
            import json
            print(email)
            # First get the current user document
            response = requests.get(
    f"http://localhost:5984/users/_design/users/_view/userdetails",
    params={"key": json.dumps(email)},
    auth=("admin", "admin123")
)
            print(response)
            if response.status_code != 200:
                return jsonify({"message": "User not found"}), 404
            data = response.json()
            rows = data.get('rows', [])
            if not rows:
                return jsonify({"message": "User not found"}), 404

            # Step 2: Get the document
            doc = rows[0]['value']  # Assuming the view emits the whole doc as value
            doc_id = doc['_id']
            doc_rev = doc['_rev']   
            user_data = response.json()
            
            # Update password
            doc['password'] = new_password
            
            # Save updated document
            update_response = requests.put(
    f"http://localhost:5984/users/{doc_id}",
    auth=("admin", "admin123"),
    headers={"Content-Type": "application/json"},
    data=json.dumps(doc)
)
            
            if update_response.status_code in [200, 201]:
                # Password updated successfully
                # Remove reset token
              
                
                return jsonify({"message": "Password reset successfully"}), 200
            else:
                return jsonify({"message": "Failed to update password"}), 500
                
        except Exception as e:
            print(f"Database error: {str(e)}")
            return jsonify({"message": "Server connection error"}), 500
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"message": "An unexpected error occurred"}), 500
@app.route('/')
def form():
    return render_template('index.html')

@app.route('/signin')
def sign_up():
    return render_template('sign-up.html')

@app.route('/login')
def log_in():
    return render_template('login.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/service')
@jwt_required()
def service():
    # This route is now protected and requires a valid token
    return render_template('services.html')

@app.route('/forget')
def forget():
    return render_template('forgetpass.html')

@app.route('/problem')
@jwt_required()  # Protecting this route too
def problem():
    return render_template('problem.html')

# New login API endpoint for JWT authentication
@app.route('/api/login', methods=['POST'])
def api_login():
    try:
        data = request.get_json()
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({"error": "Username and password required"}), 400
        
        # CouchDB authentication (your existing code)
        response = None
        try:
            import requests
            response = requests.get(
                f"http://localhost:5984/users/{username}", 
                auth=("admin", "admin123")
            )
            
            if response.status_code == 200:
                user_data = response.json()
                if user_data.get('password') == password:
                    # Create access token with identity
                    access_token = create_access_token(identity=username)
                    
                    # Create response
                    resp = jsonify({"success": True, "message": "Login successful"})
                    
                    # Set the JWT cookies in the response
                    set_access_cookies(resp, access_token)
                    
                    return resp
                else:
                    return jsonify({"error": "Incorrect password"}), 401
            else:
                return jsonify({"error": "User not found"}), 404
                
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Logout route to revoke tokens
@app.route('/logout')
def logout():
    # Get JWT identity
    try:
        # Create response
        resp = make_response(redirect(url_for('log_in')))
        
        # Unset the JWT cookies
        unset_jwt_cookies(resp)
        
        return resp
    except:
        # If there's no valid token, just redirect
        return redirect(url_for('log_in'))

@app.route('/set_condition', methods=['POST'])
@jwt_required()  # Protect this route
def set_condition():
    global selected_condition
    condition = request.form.get("pbutton")
    print(f"🔹 Received condition request with data: {request.form}")
    
    if condition:
        selected_condition = condition
        print(f"✅ Updated condition to: {selected_condition}")
        # Start camera after condition is set
        open_camera(selected_condition)
        return jsonify({"status": "success", "selected": condition})
    else:
        print("❌ No condition received in request")
        return jsonify({"status": "error", "message": "No condition received"}), 400

camera_open=False
def open_camera(condition):
    global camera_open
    if camera_open:
        print("Camera already open. Only one condition allowed at a time.")
        return "Camera already open"
    
    print(f"Opening camera with condition: {condition}")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera with index 0, trying index 1")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("Error: Could not open camera with index 1 either.")
            return
    camera_open = True

    # Create a fullscreen window
    window_name = 'Hand Tracking'
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        while cap.isOpened():
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("Window closed by user")
                break
                
            success, frame = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
                
            # Process frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = hands.process(frame_rgb)
            frame_rgb.flags.writeable = True
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    if is_hand_in_range(hand_landmarks) and is_hand_straight(hand_landmarks):
                        if condition:
                            gesture_recognition(frame, hand_landmarks, condition)
                            cv2.putText(frame, f"Active: {condition}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "Hand is not in range or not straight", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow(window_name, frame)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)  # Keep on top
            
            # Wait for a short time, but don't rely on 'q' to exit
            if cv2.waitKey(5) & 0xFF == ord('q'):  # Use ESC as backup exit method
                camera_open = False
                break
            

        cap.release()
        cv2.destroyAllWindows()
        # camera_open = False
# Entry point
if __name__ == '__main__':
    # Start the Flask server
    print("Starting Flask server on http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=True)  # Set debug to False when running with camera