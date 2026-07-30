import os
import re
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session
from groq import Groq
from dotenv import load_dotenv

# Supabase SDK Import
try:
    from supabase import create_client, Client
except ImportError:
    create_client = None
    Client = None

# Load workspace environment variables
load_dotenv()

# Resolve absolute template path configuration for robust serverless deployments
base_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(base_dir, '..', 'templates')

app = Flask(__name__, template_folder=template_dir)

# Set application session key configuration securely
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "safesphere-secret-key-2026")

# API & Auth Credentials
SUPABASE_URL = os.environ.get("SUPABASE_URL")
# IMPORTANT: this must be the Supabase SERVICE ROLE key (not the anon/public key).
# The backend is the trust boundary here -- every route below checks
# session['user_id'] itself before touching Supabase, so the service role key
# is used to read/write on the user's behalf. Never expose this key to the
# frontend; it only ever lives in this server-side file/env var.
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_KEY")
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")

# Initialize Supabase client securely
supabase_client = None
if create_client and SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        print(f"Supabase Client Initialization Notice: {e}")

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


def calculate_safety_score(density, hazard):
    """
    Calculates a baseline mathematical Community Safety Score (out of 100,
    higher = safer) from reported area density and primary hazard concern,
    to give users an engaging, immediate personal risk indicator.

    NOTE: the `density` parameter is persisted under the legacy
    'usage_level' column name in Supabase to avoid a schema migration.
    """
    base_score = 100

    density_normalized = density.lower().strip()
    if density_normalized == 'urban':
        base_score -= 25
    elif density_normalized == 'suburban':
        base_score -= 12
    elif density_normalized == 'rural':
        base_score -= 18  # slower emergency response times in remote areas

    hazard_normalized = hazard.lower().strip()
    hazard_penalties = {
        'crime': 20,
        'fire': 15,
        'storm & flooding': 15,
        'severe winter weather': 10
    }

    penalty = hazard_penalties.get(hazard_normalized, 8)
    final_score = max(10, base_score - penalty)
    return final_score


def supabase_required():
    return supabase_client is not None


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
    if not supabase_required():
        return jsonify({"error": "Supabase is not configured on the server (missing SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY)."}), 500

    data = request.get_json() or {}
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()

    if not username or not password:
        return jsonify({"error": "Username and password are required parameters."}), 400

    email = username if "@" in username else f"{username}@safesphere.app"

    try:
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
            return jsonify({"success": True, "message": "Account created successfully. You can now sign in."})
        return jsonify({"error": "Supabase did not return a user for this sign-up."}), 500
    except Exception as e:
        return jsonify({"error": f"Supabase registration error: {str(e)}"}), 400


@app.route('/api/login', methods=['POST'])
def login():
    if not supabase_required():
        return jsonify({"error": "Supabase is not configured on the server (missing SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY)."}), 500

    data = request.get_json() or {}
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()

    if not username or not password:
        return jsonify({"error": "Username and password are required parameters."}), 400

    email = username if "@" in username else f"{username}@safesphere.app"

    try:
        res = supabase_client.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        if res and res.user:
            display_username = (res.user.user_metadata or {}).get('username', username)
            session['user_id'] = res.user.id
            session['username'] = display_username
            session['auth_provider'] = 'supabase'
            return jsonify({"success": True, "username": display_username, "provider": "supabase"})
        return jsonify({"error": "Invalid credential pair configuration."}), 401
    except Exception as e:
        return jsonify({"error": f"Invalid credential pair configuration: {str(e)}"}), 401


@app.route('/api/google-login', methods=['POST'])
def google_login():
    """Authenticates users via Supabase's native Google ID token sign-in.

    Requires the Google provider to be enabled in Supabase Auth settings, with
    the same Google OAuth Client ID configured there and used on the frontend.
    """
    if not supabase_required():
        return jsonify({"error": "Supabase is not configured on the server (missing SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY)."}), 500

    data = request.get_json() or {}
    token = data.get('id_token') or data.get('credential')

    if not token:
        return jsonify({"error": "Google ID token credential is required."}), 400

    try:
        res = supabase_client.auth.sign_in_with_id_token({
            "provider": "google",
            "token": token
        })
        if not res or not res.user:
            return jsonify({"error": "Invalid or expired Google token credential."}), 401

        email = res.user.email
        metadata = res.user.user_metadata or {}
        username = metadata.get('username') or metadata.get('full_name') or metadata.get('name') or (email.split('@')[0] if email else 'Google User')

        session['user_id'] = res.user.id
        session['username'] = username
        session['email'] = email
        session['auth_provider'] = 'google'

        return jsonify({
            "success": True,
            "username": username,
            "email": email,
            "provider": "google"
        })
    except Exception as e:
        return jsonify({"error": f"Failed to verify Google Token: {str(e)}"}), 401


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
            "provider": session.get('auth_provider', 'supabase')
        })
    return jsonify({"logged_in": False})


@app.route('/api/history', methods=['GET'])
def get_history():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized endpoint access. User must log in first."}), 401

    if not supabase_required():
        return jsonify({"error": "Supabase is not configured on the server."}), 500

    try:
        res = (
            supabase_client.table('assessments')
            .select('zip_code, usage_level, concern, stewardship_score, analysis, created_at')
            .eq('user_id', session['user_id'])
            .order('created_at', desc=True)
            .execute()
        )
        rows = res.data or []
        history_list = [
            {
                "zip_code": row["zip_code"],
                "density": row["usage_level"],
                "hazard": row["concern"],
                "safety_score": row["stewardship_score"],
                "analysis": row["analysis"],
                "created_at": row["created_at"]
            }
            for row in rows
        ]
        return jsonify({"success": True, "history": history_list})
    except Exception as e:
        return jsonify({"error": f"Failed to load assessment history: {str(e)}"}), 500


@app.route('/api/evaluate', methods=['POST'])
def evaluate_region():
    """
    Mode 1: Risk Assessment Module
    Ingests local metrics, runs US spatial validation, calculates a community safety score,
    and returns a structured AI risk evaluation analyzing local hazard and safety conditions.
    """
    if not client:
        return jsonify({"error": "Groq API token configuration is missing on the server environment."}), 500

    data = request.get_json() or {}
    zip_code = data.get('zip_code', '').strip()
    density = data.get('usage', '').strip()
    hazard = data.get('concern', '').strip()

    if not zip_code or not validate_us_zip(zip_code):
        return jsonify({"error": "Invalid location context. Please provide a valid 5-digit US Zip Code."}), 400

    if not density or not hazard:
        return jsonify({"error": "Missing essential parameters. Area density and primary hazard are required."}), 400

    safety_score = calculate_safety_score(density, hazard)

    system_instruction = (
        "You are SafeSphere, an expert safety & emergency risk analysis engine tailored for the US. "
        "Your goal is to provide a comprehensive Risk Assessment analyzing local hazard exposure, "
        "neighborhood safety vulnerabilities, and practical protective measures. "
        "Align assessments with standard safety framings inspired by FEMA and American Red Cross advisory frameworks. "
        "Structure your response elegantly using clear Markdown headers, bold accents, and distinct spacing."
    )

    user_query = (
        f"Analyze this US regional hazard snapshot and compile a community safety profile:\n"
        f"- Target US Zip Code Region: {zip_code}\n"
        f"- Area & Population Density: {density}\n"
        f"- Primary Hazard Concern: {hazard}\n\n"
        f"Output structural guidelines addressing:\n"
        f"1. A localized 'Regional Risk Assessment' analyzing hazard exposure and safety vulnerabilities.\n"
        f"2. Explicit 'Contextual Safety Alerts' highlighting common risk indicators and warning signs.\n"
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

        # Save evaluation to Supabase assessment history if user session is active.
        # NOTE: density/hazard/safety_score are persisted under the legacy
        # usage_level/concern/stewardship_score column names to avoid a schema migration.
        if 'user_id' in session and supabase_required():
            try:
                supabase_client.table('assessments').insert({
                    "user_id": session['user_id'],
                    "zip_code": zip_code,
                    "usage_level": density,
                    "concern": hazard,
                    "stewardship_score": safety_score,
                    "analysis": analysis_payload,
                    "created_at": datetime.utcnow().isoformat()
                }).execute()
            except Exception as e:
                print(f"Assessment history save skipped: {e}")

        return jsonify({
            "success": True,
            "safety_score": safety_score,
            "analysis": analysis_payload
        })

    except Exception as e:
        return jsonify({"error": f"Groq processing pipeline hit an execution fault: {str(e)}"}), 500


@app.route('/api/chat', methods=['POST'])
def chat_advisory():
    """
    Mode 2: Safety Advisory Chat Hub
    """
    if not client:
        return jsonify({"error": "Groq API token configuration is missing on the server environment."}), 500

    data = request.get_json() or {}
    user_message = data.get('message', '').strip()
    chat_history = data.get('history', [])

    if not user_message:
        return jsonify({"error": "Chat message context cannot be blank."}), 400

    system_instruction = (
        "You are the SafeSphere Safety Advisory Chat Hub. You act as an interactive personal safety advisor. "
        "Provide immediate, step-by-step home safety, personal safety, and disaster preparedness guidance. "
        "Ensure all guidance assumes a US municipal context (e.g., standard American Red Cross emergency kits). "
        "Keep replies highly operational, punchy, concise, and structured with clean bullet points."
    )

    messages = [{"role": "system", "content": system_instruction}]

    for turn in chat_history[-6:]:
        if isinstance(turn, dict) and 'role' in turn and 'content' in turn:
            messages.append({"role": turn['role'], "content": turn['content']})

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


@app.route('/api/emergency', methods=['POST'])
def emergency_assistance():
    """
    Mode 3: AI Emergency Assistance
    Analyzes natural language descriptions of emergency situations and provides structured guidance.
    """
    if not client:
        return jsonify({"error": "Groq API token configuration is missing on the server environment."}), 500

    data = request.get_json() or {}
    description = data.get('description', '').strip() or data.get('situation', '').strip()

    if not description:
        return jsonify({"error": "Emergency situation description is required."}), 400

    system_instruction = (
        "You are SafeSphere Emergency Assistant, an AI first-response and safety guidance coordinator. "
        "Your objective is to analyze natural language descriptions of emergencies or dangerous situations "
        "and provide immediate, highly structured, clear guidance prioritizing life safety.\n\n"
        "REQUIRED STRUCTURE & CONTENT:\n"
        "1. ## Situation Assessment\n"
        "   - A brief, calm assessment of the reported situation, identifying the emergency type and severity.\n"
        "2. ## Immediate Actions Needed\n"
        "   - Prioritized, clear, numbered steps to take immediately to secure safety.\n"
        "3. ## Basic First Aid & Safety Instructions\n"
        "   - Step-by-step first aid or technical safety procedures tailored to the hazard (e.g., CPR, hemorrhage control, burn care, shelter protocols, hazard isolation).\n"
        "4. ## Warning Signs & Red Flags\n"
        "   - Clear symptoms or physical/environmental signs indicating urgent need for professional emergency services.\n"
        "5. ## Emergency Response & 911 Recommendation\n"
        "   - Prominent directive to call 911 or local emergency responders immediately if the situation is life-threatening or escalates.\n"
        "6. ## Additional Safety & Harm Prevention Tips\n"
        "   - Practical measures to prevent further injury, exposure, or danger while awaiting first responders.\n\n"
        "SAFETY MANDATE:\n"
        "Explicitly state in the response that this guidance supports emergency preparedness and immediate decision-making, "
        "and IS NOT a substitute for professional medical care, fire services, or official emergency responders."
    )

    user_query = f"EMERGENCY REPORT / DESCRIPTION:\n{description}"

    try:
        completion = client.chat.completions.create(
            model=MODEL_EVALUATOR,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_query}
            ],
            temperature=0.2,
            max_tokens=1000
        )

        guidance_payload = completion.choices[0].message.content

        return jsonify({
            "success": True,
            "guidance": guidance_payload
        })

    except Exception as e:
        return jsonify({"error": f"Emergency assistance engine hit an execution fault: {str(e)}"}), 500


@app.route('/api/planner', methods=['POST'])
def preparedness_planner():
    """
    Mode 4: Emergency Preparedness Planner
    Generates customized emergency preparedness plans including supply checklists,
    evacuation plans, communication plans, and pet preparedness recommendations.
    """
    if not client:
        return jsonify({"error": "Groq API token configuration is missing on the server environment."}), 500

    data = request.get_json() or {}
    zip_code = data.get('zip_code', '').strip()
    household_size = data.get('household_size', '1').strip()
    has_pets = data.get('has_pets', False)
    special_needs = data.get('special_needs', '').strip()
    primary_hazards = data.get('primary_hazards', '').strip()

    if not zip_code or not validate_us_zip(zip_code):
        return jsonify({"error": "Valid 5-digit US Zip Code required for planner location context."}), 400

    system_instruction = (
        "You are SafeSphere Emergency Preparedness Planner, an expert disaster resilience coordinator. "
        "Your goal is to generate tailored, actionable emergency preparedness plans based on household needs and local geographic context.\n\n"
        "REQUIRED STRUCTURE:\n"
        "1. ## Custom Emergency Supply Checklist\n"
        "   - Detailed checklist of essential supply items, water/food quantities scaled to household size, first aid, power, hygiene, and tools.\n"
        "2. ## Family Evacuation & Action Plan\n"
        "   - Step-by-step evacuation protocols, primary and secondary exit routes, meeting locations, and hazard safety rules.\n"
        "3. ## Family Communication Plan\n"
        "   - Instructions for out-of-area emergency contacts, emergency signal methods, and local emergency alert subscription guidance.\n"
        "4. ## Pet & Special Household Needs Protocols\n"
        "   - Targeted recommendations for pet survival kits and evacuation, or medical/mobility/infant/elderly considerations if applicable.\n"
        "5. ## Regional Hazard Mitigation Advice\n"
        "   - Specific localized preparation advice tailored to the region and hazards specified.\n\n"
        "SAFETY MANDATE:\n"
        "Remind users that this plan provides structured preparation guidance aligned with standard emergency planning frameworks (FEMA/Red Cross) and should be reviewed regularly with household members."
    )

    user_query = (
        f"Generate a customized Emergency Preparedness Plan for:\n"
        f"- Location ZIP Code: {zip_code}\n"
        f"- Household Size: {household_size} person(s)\n"
        f"- Pet Accommodations Needed: {'Yes' if has_pets else 'No'}\n"
        f"- Special Medical / Mobility / Family Needs: {special_needs if special_needs else 'None specified'}\n"
        f"- Primary Hazard Focus: {primary_hazards if primary_hazards else 'General emergency preparedness'}"
    )

    try:
        completion = client.chat.completions.create(
            model=MODEL_EVALUATOR,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_query}
            ],
            temperature=0.3,
            max_tokens=1000
        )

        plan_payload = completion.choices[0].message.content

        return jsonify({
            "success": True,
            "plan": plan_payload
        })

    except Exception as e:
        return jsonify({"error": f"Preparedness planner engine hit an execution fault: {str(e)}"}), 500


@app.route('/api/knowledge', methods=['POST'])
def safety_knowledge():
    """
    Mode 5: Safety Knowledge Center
    AI-powered educational Q&A resource for first aid, disaster prep, severe weather, home safety, and outdoor safety.
    """
    if not client:
        return jsonify({"error": "Groq API token configuration is missing on the server environment."}), 500

    data = request.get_json() or {}
    question = data.get('question', '').strip()
    category = data.get('category', 'General Safety').strip()

    if not question:
        return jsonify({"error": "Question context cannot be blank."}), 400

    system_instruction = (
        "You are the SafeSphere Safety Knowledge Center, an authoritative, accessible educational AI resource for emergency preparedness, first aid, disaster response, severe weather, home safety, and outdoor safety.\n\n"
        "REQUIRED GUIDELINES:\n"
        "- Provide clear, reliable, easy-to-understand explanations and practical safety advice.\n"
        "- Structure responses with clean Markdown formatting, bold headings, bullet points, and actionable tips.\n"
        "- State clearly that information is for educational purposes and is not a substitute for professional emergency services or medical personnel."
    )

    user_query = f"Category: {category}\nQuestion: {question}"

    try:
        completion = client.chat.completions.create(
            model=MODEL_CHAT,
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_query}
            ],
            temperature=0.4,
            max_tokens=800
        )

        answer_payload = completion.choices[0].message.content

        return jsonify({
            "success": True,
            "answer": answer_payload
        })

    except Exception as e:
        return jsonify({"error": f"Safety knowledge engine hit an execution fault: {str(e)}"}), 500


@app.errorhandler(404)
def resource_not_found(e):
    return jsonify({"error": "The specified route configuration does not exist on this application."}), 404


if __name__ == '__main__':
    app.run(debug=True)
