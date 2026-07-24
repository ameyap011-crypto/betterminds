import os
import re
import sqlite3
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session, g
from groq import Groq
from dotenv import load_dotenv
from werkzeug.security import generate_password_hash, check_password_hash
import requests

# Supabase SDK Import
try:
    from supabase import create_client, Client
except ImportError:
    create_client = None
    Client = None

# Google OAuth Import
try:
    from google.oauth2 import id_token
    from google.auth.transport import requests as google_requests
except ImportError:
    id_token = None
    google_requests = None

# Load workspace environment variables
load_dotenv()

# Resolve absolute template path configuration for robust serverless deployments
base_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(base_dir, '..', 'templates')

app = Flask(__name__, template_folder=template_dir)

# Set application session key configuration securely
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "sentinelai-secret-key-2026")

DATABASE = os.path.join(base_dir, 'sentinelai.db')

# API & Auth Credentials
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")

# Initialize Supabase client securely
supabase_client = None
if create_client and SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        print(f"Supabase Client Initialization Notice: {e}")

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db():
    with app.app_context():
        db = get_db()
        cursor = db.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                zip_code TEXT NOT NULL,
                usage_level TEXT NOT NULL,
                concern TEXT NOT NULL,
                stewardship_score INTEGER NOT NULL,
                analysis TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        ''')
        db.commit()

# Initialize schema definitions upon runtime allocation
init_db()

# Initialize the Groq Engine Client securely
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = None
if GROQ_API_KEY:
    client = Groq(api_key=GROQ_API_KEY)

# Define verified, active, non-decommissioned Groq model identifiers for 2026
MODEL_EVALUATOR = "openai/gpt-oss-120b"  # High reasoning capability for detailed risk metrics
MODEL_CHAT = "openai/gpt-oss-20b"        # Sub-second token delivery optimized for user chat loops

def validate_us_zip(zip_code):
    """
    Enforces the geographical boundary constraint.
    Validates if the submitted value follows a clean 5-digit US Zip Code format.
    """
    return bool(re.match(r"^\d{5}$", str(zip_code).strip()))

def calculate_stewardship_score(usage, concern):
    """
    Calculates a baseline mathematical Eco-Stewardship Score (out of 100)
    to transform abstract user habits into an engaging personal indicator.
    """
    base_score = 100
    
    # Evaluate estimated resource usage footprint
    usage_normalized = usage.lower().strip()
    if usage_normalized == 'high':
        base_score -= 40
    elif usage_normalized == 'moderate':
        base_score -= 20
    elif usage_normalized == 'low':
        base_score -= 5
        
    # Apply context weight adjustments based on primary environmental concern
    concern_normalized = concern.lower().strip()
    concern_penalties = {
        'drought': 15,
        'heatwaves': 10,
        'clean water access': 20,
        'severe weather': 10
    }
    
    penalty = concern_penalties.get(concern_normalized, 5)
    final_score = max(10, base_score - penalty)
    return final_score

@app.route('/')
def index():
    """Renders the core single-page application dashboard user interface."""
    return render_template('index.html')

@app.route('/api/config', methods=['GET'])
def get_config():
    """Returns client configuration keys for Google Sign-In and Supabase."""
    return jsonify({
        "google_client_id": GOOGLE_CLIENT_ID or "",
        "supabase_url": SUPABASE_URL or ""
    })

@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json() or {}
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()
    
    if not username or not password:
        return jsonify({"error": "Username and password are required parameters."}), 400

    supabase_created = False
    supabase_err_msg = None

    # Attempt Supabase Authentication Sign Up if Supabase is configured
    if supabase_client:
        try:
            email = username if "@" in username else f"{username}@sentinelai.app"
            res = supabase_client.auth.sign_up({
                "email": email,
                "password": password,
                "options": {
                    "data": {
                        "username": username
                    }
                }
            })
            if res and res.user:
                supabase_created = True
        except Exception as e:
            supabase_err_msg = str(e)

    # Local DB Registration (maintains application database relations)
    db = get_db()
    cursor = db.cursor()
    try:
        hashed_password = generate_password_hash(password)
        cursor.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, hashed_password))
        db.commit()
        return jsonify({"success": True, "message": "Account created successfully via Supabase & Local DB."})
    except sqlite3.IntegrityError:
        if supabase_created:
            return jsonify({"success": True, "message": "Account created successfully via Supabase."})
        return jsonify({"error": "Username is already registered in our system."}), 400
    except Exception as e:
        if supabase_created:
            return jsonify({"success": True, "message": "Account created successfully via Supabase."})
        return jsonify({"error": f"Database error: {supabase_err_msg or str(e)}"}), 500

@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json() or {}
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()
    
    if not username or not password:
        return jsonify({"error": "Username and password are required parameters."}), 400

    # 1. Attempt authentication with Supabase Auth
    if supabase_client:
        try:
            email = username if "@" in username else f"{username}@sentinelai.app"
            res = supabase_client.auth.sign_in_with_password({
                "email": email,
                "password": password
            })
            if res and res.user:
                db = get_db()
                cursor = db.cursor()
                cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
                row = cursor.fetchone()
                if row:
                    user_id = row['id']
                else:
                    cursor.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", 
                                   (username, generate_password_hash(password)))
                    db.commit()
                    user_id = cursor.lastrowid

                session['user_id'] = user_id
                session['username'] = username
                session['auth_provider'] = 'supabase'
                return jsonify({"success": True, "username": username, "provider": "supabase"})
        except Exception as e:
            print(f"Supabase Login attempt skipped to fallback: {e}")

    # 2. Local DB Authentication Fallback
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cursor.fetchone()
    
    if user and check_password_hash(user['password_hash'], password):
        session['user_id'] = user['id']
        session['username'] = user['username']
        session['auth_provider'] = 'local'
        return jsonify({"success": True, "username": user['username'], "provider": "local"})
    else:
        return jsonify({"error": "Invalid credential pair configuration."}), 401

@app.route('/api/google-login', methods=['POST'])
def google_login():
    """Authenticates users via Google Cloud Console OAuth ID Tokens."""
    data = request.get_json() or {}
    token = data.get('id_token') or data.get('credential')
    
    if not token:
        return jsonify({"error": "Google ID token credential is required."}), 400

    google_user_id = None
    email = None
    name = None
    verified = False

    # Attempt verification with python google-auth library
    if id_token and google_requests:
        try:
            req = google_requests.Request()
            id_info = id_token.verify_oauth2_token(token, req, GOOGLE_CLIENT_ID if GOOGLE_CLIENT_ID else None)
            google_user_id = id_info.get('sub')
            email = id_info.get('email')
            name = id_info.get('name') or (email.split('@')[0] if email else 'Google User')
            verified = True
        except Exception as e:
            print(f"Google Token Verification via google-auth failed: {e}")

    # Fallback verification with Google's public tokeninfo endpoint
    if not verified:
        try:
            resp = requests.get(f"https://oauth2.googleapis.com/tokeninfo?id_token={token}", timeout=5)
            if resp.status_code == 200:
                id_info = resp.json()
                google_user_id = id_info.get('sub')
                email = id_info.get('email')
                name = id_info.get('name') or (email.split('@')[0] if email else 'Google User')
                verified = True
        except Exception as e:
            return jsonify({"error": f"Failed to verify Google Token: {str(e)}"}), 401

    if not verified or not google_user_id:
        return jsonify({"error": "Invalid or expired Google token credential."}), 401

    # Sync Google User to local SQLite DB for relational integrity in assessments
    db = get_db()
    cursor = db.cursor()
    username = name or (email.split('@')[0] if email else f"google_user_{google_user_id[:6]}")
    
    cursor.execute("SELECT id, username FROM users WHERE username = ?", (username,))
    row = cursor.fetchone()
    if row:
        user_id = row['id']
    else:
        cursor.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", 
                       (username, generate_password_hash(os.urandom(16).hex())))
        db.commit()
        user_id = cursor.lastrowid

    session['user_id'] = user_id
    session['username'] = username
    session['email'] = email
    session['auth_provider'] = 'google'

    return jsonify({
        "success": True, 
        "username": username, 
        "email": email, 
        "provider": "google"
    })

@app.route('/api/logout', methods=['POST', 'GET'])
def logout():
    session.pop('user_id', None)
    session.pop('username', None)
    session.pop('email', None)
    session.pop('auth_provider', None)
    return jsonify({"success": True, "message": "Session terminated successfully."})

@app.route('/api/user-status', methods=['GET'])
def user_status():
    if 'user_id' in session:
        return jsonify({
            "logged_in": True, 
            "username": session['username'],
            "provider": session.get('auth_provider', 'local')
        })
    return jsonify({"logged_in": False})

@app.route('/api/history', methods=['GET'])
def get_history():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized endpoint access. User must log in first."}), 401
        
    db = get_db()
    cursor = db.cursor()
    cursor.execute(
        "SELECT zip_code, usage_level, concern, stewardship_score, analysis, created_at FROM assessments WHERE user_id = ? ORDER BY created_at DESC",
        (session['user_id'],)
    )
    rows = cursor.fetchall()
    
    history_list = []
    for row in rows:
        history_list.append({
            "zip_code": row["zip_code"],
            "usage": row["usage_level"],
            "concern": row["concern"],
            "stewardship_score": row["stewardship_score"],
            "analysis": row["analysis"],
            "created_at": row["created_at"]
        })
        
    return jsonify({"success": True, "history": history_list})

@app.route('/api/evaluate', methods=['POST'])
def evaluate_region():
    """
    Mode 1: Regional Profile & Threat Evaluator
    Ingests local metrics, runs US spatial validation, calculates the environmental
    impact score, and hits Groq for an educational resource assessment.
    """
    if not client:
        return jsonify({"error": "Groq API token configuration is missing on the server environment."}), 500

    data = request.get_json() or {}
    zip_code = data.get('zip_code', '').strip()
    usage = data.get('usage', '').strip()
    concern = data.get('concern', '').strip()

    # Apply strict US boundary guardrail checks
    if not zip_code or not validate_us_zip(zip_code):
        return jsonify({"error": "Invalid location context. Please provide a valid 5-digit US Zip Code."}), 400

    if not usage or not concern:
        return jsonify({"error": "Missing essential parameters. Usage level and primary concern are required."}), 400

    # Execute backend impact score evaluation
    stewardship_score = calculate_stewardship_score(usage, concern)

    # Engineering system instructions tailored specifically for US open eco-data standards
    system_instruction = (
        "You are SentinelAI, an expert humanitarian environmental analysis engine tailored for the US environment. "
        "Your goal is to provide educational regional risk assessments and neighborhood conservation tips. "
        "Align assessments with standard safety framings inspired by FEMA and EPA advisory frameworks. "
        "Structure your response elegantly using clear Markdown headers, bold accents, and distinct spacing."
    )
    
    user_query = (
        f"Analyze this US regional sustainability snapshot and compile a community profile:\n"
        f"- Target US Zip Code Region: {zip_code}\n"
        f"- Reported Household Resource Footprint: {usage}\n"
        f"- Target Local Safety & Resource Crisis Parameter: {concern}\n\n"
        f"Output structural guidelines addressing:\n"
        f"1. A localized 'Regional Resource Assessment' linking resource usage habits to regional environmental limits.\n"
        f"2. Explicit 'Contextual Safety Alerts' highlighting common indicators of resource vulnerability.\n"
        f"3. Three actionable, step-by-step community-level mitigation ideas for disaster resilience."
    )

    try:
        completion = client.chat.completions.create(
            model=MODEL_EVALUATOR,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_query}
            ],
            temperature=0.25,
            max_tokens=900
        )
        
        analysis_payload = completion.choices[0].message.content
        
        # Save evaluation to assessment history database if user session is active
        if 'user_id' in session:
            try:
                db = get_db()
                cursor = db.cursor()
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cursor.execute(
                    "INSERT INTO assessments (user_id, zip_code, usage_level, concern, stewardship_score, analysis, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (session['user_id'], zip_code, usage, concern, stewardship_score, analysis_payload, current_time)
                )
                db.commit()
            except Exception:
                pass

        return jsonify({
            "success": True,
            "stewardship_score": stewardship_score,
            "analysis": analysis_payload
        })

    except Exception as e:
        return jsonify({"error": f"Groq processing pipeline hit an execution fault: {str(e)}"}), 500

@app.route('/api/chat', methods=['POST'])
def chat_advisory():
    """
    Mode 2: Eco-Safety Advisory Chat Hub
    Evaluates dynamic incoming chat interactions regarding local conservation and neighborhood
    preparedness with fast token completion times.
    """
    if not client:
        return jsonify({"error": "Groq API token configuration is missing on the server environment."}), 500

    data = request.get_json() or {}
    user_message = data.get('message', '').strip()
    chat_history = data.get('history', [])

    if not user_message:
        return jsonify({"error": "Chat message context cannot be blank."}), 400

    system_instruction = (
        "You are the SentinelAI Eco-Safety Advisory Chat Hub. You act as an interactive neighborhood safety monitor. "
        "Provide immediate, step-by-step micro-level resource saving and family preparation strategies. "
        "Ensure all guidance assumes a US municipal context (e.g., standard American Red Cross emergency kits). "
        "Keep replies highly operational, punchy, concise, and structured with clean bullet points."
    )

    # Initialize message array with structural system parameters
    messages = [{"role": "system", "content": system_instruction}]
    
    # Append slice of history data to preserve context without blowing up execution tokens
    for turn in chat_history[-6:]:
        if isinstance(turn, dict) and 'role' in turn and 'content' in turn:
            messages.append({"role": turn['role'], "content": turn['content']})
            
    # Append the fresh user communication payload
    messages.append({"role": "user", "content": user_message})

    try:
        completion = client.chat.completions.create(
            model=MODEL_CHAT,
            messages=messages,
            temperature=0.55,
            max_tokens=600
        )
        
        reply_payload = completion.choices[0].message.content
        
        return jsonify({
            "success": True,
            "reply": reply_payload
        })

    except Exception as e:
        return jsonify({"error": f"Chat integration engine hit an execution fault: {str(e)}"}), 500

@app.errorhandler(404)
def resource_not_found(e):
    return jsonify({"error": "The specified route configuration does not exist on this application."}), 404

if __name__ == '__main__':
    app.run(debug=True)