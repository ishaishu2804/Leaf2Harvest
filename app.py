from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from app.utils.crop_predictor import recommend_crop
from app.utils.weather_fetcher import get_weather, get_weather_data, assess_pest_risk, assess_5day_risk
import joblib
import numpy as np
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.models import Base, Product, User, Order, OrderItem, Diagnosis, WeatherData, PestRiskData, DiseaseData, UsageLog, ApiUsageLog
import datetime
from werkzeug.utils import secure_filename
import tensorflow as tf
from PIL import Image
import io
import json # Import json
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    openai = None
    OPENAI_AVAILABLE = False
import base64

# Import new services
from app.services.openai_service import get_openai_service
from app.utils.usage_logger import UsageLogger



# Potentially needed for yield forecasting model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor # Example model type
from sklearn.metrics import mean_squared_error # Example evaluation metric

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'app', 'templates')
STATIC_DIR = os.path.join(BASE_DIR, 'app', 'static') # Define static folder

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
app.config['SECRET_KEY'] = 'your_secret_key_here' # Replace with a real secret key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///your_database.db?check_same_thread=False'

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Get the absolute path to the models directory and database
MODELS_DIR = os.path.join(BASE_DIR, 'app', 'models')
DATABASE_PATH = os.path.join(BASE_DIR, 'app', 'site.db') # Correct path to the database

# Set up database engine and session
engine = create_engine(f'sqlite:///{DATABASE_PATH}?check_same_thread=False')
Base.metadata.create_all(engine)  # Create all tables
Base.metadata.bind = engine

DBSession = sessionmaker(bind=engine)
session = DBSession()

# Initialize usage logger
usage_logger = UsageLogger(f'sqlite:///{DATABASE_PATH}?check_same_thread=False')

# Load model and encoder
# Ensure these files exist before running the app
try:
    pest_model = joblib.load(os.path.join(MODELS_DIR, 'pest_model.pkl'))
    crop_encoder = joblib.load(os.path.join(MODELS_DIR, 'crop_encoder.pkl'))
except FileNotFoundError as e:
    print(f"[ERROR] Pest model or encoder file not found: {e}. Please run the training scripts first.")
    pest_model = None # Set to None to avoid app crashing
    crop_encoder = None

# Load the disease detection model and class names (already here)
try:
    disease_model_path = os.path.join(MODELS_DIR, 'disease_model.h5')
    if os.path.exists(disease_model_path):
        disease_model = tf.keras.models.load_model(disease_model_path)
        print("[INFO] Disease model loaded successfully.")
    else:
        print(f"[WARNING] Disease model file not found at {disease_model_path}")
        print("[INFO] To train the disease model, run: python app/models/train_disease_model.py")
        disease_model = None
    
    # Load class names
    class_names_path = os.path.join(MODELS_DIR, 'class_names.json')
    if os.path.exists(class_names_path):
        with open(class_names_path, 'r') as f:
            disease_class_names = json.load(f)
        print("[INFO] Disease class names loaded successfully.")
    else:
        print(f"[WARNING] Class names file not found at {class_names_path}")
        disease_class_names = []

except Exception as e:
    print(f"[ERROR] Error loading disease model or class names: {e}")
    disease_model = None
    disease_class_names = []

# Add these configurations after app initialization
UPLOAD_FOLDER = os.path.join(STATIC_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =============================================================================
# OPENAI CONFIGURATION
# =============================================================================
# To fix the quota issue, replace the API key below with your new OpenAI API key
# Get a new key from: https://platform.openai.com/api-keys
# =============================================================================

if OPENAI_AVAILABLE:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if OPENAI_API_KEY:
        openai.api_key = OPENAI_API_KEY
        print(f"[INFO] OpenAI API configured. Key starts with: {OPENAI_API_KEY[:20]}...")
    else:
        print("[WARNING] OPENAI_API_KEY not found in environment variables.")
else:
    OPENAI_API_KEY = None
    print("[WARNING] OpenAI package not available. Using fallback analysis only.")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_treatment_recommendations(disease_name):
    """Generate treatment recommendations based on disease name"""
    treatments = {
        'healthy': "Your plant appears to be healthy! Continue with regular care including proper watering, adequate sunlight, and balanced nutrition.",
        
        'bacterial_spot': """Bacterial Spot Treatment:
1. Remove infected leaves immediately
2. Apply copper-based fungicide
3. Improve air circulation
4. Avoid overhead watering
5. Use disease-resistant varieties in future""",
        
        'early_blight': """Early Blight Treatment:
1. Remove affected leaves and debris
2. Apply fungicide containing chlorothalonil
3. Improve plant spacing for better air circulation
4. Water at soil level, not on leaves
5. Rotate crops next season""",
        
        'late_blight': """Late Blight Treatment:
1. Remove all infected plant material immediately
2. Apply fungicide with metalaxyl or copper
3. Improve drainage and air circulation
4. Avoid planting in same area next year
5. Consider resistant varieties""",
        
        'leaf_mold': """Leaf Mold Treatment:
1. Remove infected leaves
2. Apply fungicide containing chlorothalonil
3. Reduce humidity and improve ventilation
4. Space plants properly
5. Water early in the day""",
        
        'septoria_leaf_spot': """Septoria Leaf Spot Treatment:
1. Remove infected leaves and debris
2. Apply copper-based fungicide
3. Improve air circulation
4. Avoid overhead watering
5. Rotate crops annually""",
        
        'spider_mites': """Spider Mite Treatment:
1. Spray with water to dislodge mites
2. Apply insecticidal soap or neem oil
3. Increase humidity around plants
4. Introduce beneficial insects
5. Remove heavily infested leaves""",
        
        'target_spot': """Target Spot Treatment:
1. Remove infected leaves immediately
2. Apply fungicide with azoxystrobin
3. Improve air circulation
4. Avoid wetting leaves when watering
5. Use mulch to prevent soil splash""",
        
        'mosaic_virus': """Mosaic Virus Treatment:
1. Remove infected plants immediately
2. Control aphid vectors with insecticides
3. Disinfect tools between plants
4. Use virus-free seeds/plants
5. Practice good sanitation""",
        
        'yellow_leaf_curl': """Yellow Leaf Curl Treatment:
1. Remove infected plants
2. Control whitefly vectors
3. Use reflective mulches
4. Apply systemic insecticides
5. Plant resistant varieties"""
    }
    
    # Find best match
    disease_lower = disease_name.lower()
    for key, treatment in treatments.items():
        if key in disease_lower or disease_lower in key:
            return treatment
    
    # Default treatment for unknown diseases
    return """General Plant Disease Treatment:
1. Remove infected plant parts immediately
2. Apply appropriate fungicide or pesticide
3. Improve growing conditions (light, air, water)
4. Practice crop rotation
5. Consult with local agricultural extension service
6. Consider resistant varieties for future planting"""

def parse_openai_response(response_text, crop_type):
    """Parse OpenAI response into structured format"""
    try:
        lines = response_text.split('\n')
        disease = "Unknown Disease"
        severity = "Medium"
        symptoms = "Various symptoms detected"
        treatment = "General treatment recommendations"
        prevention = "General prevention measures"
        
        for line in lines:
            line = line.strip()
            if line.startswith('DISEASE:'):
                disease = line.replace('DISEASE:', '').strip()
            elif line.startswith('SEVERITY:'):
                severity = line.replace('SEVERITY:', '').strip()
            elif line.startswith('SYMPTOMS:'):
                symptoms = line.replace('SYMPTOMS:', '').strip()
            elif line.startswith('TREATMENT:'):
                treatment = line.replace('TREATMENT:', '').strip()
            elif line.startswith('PREVENTION:'):
                prevention = line.replace('PREVENTION:', '').strip()
        
        # Calculate confidence based on severity
        severity_scores = {'Low': 60, 'Medium': 75, 'High': 90}
        confidence = severity_scores.get(severity, 75)
        
        # Boost confidence for specific disease mentions
        if any(keyword in disease.lower() for keyword in ['blight', 'spot', 'mold', 'rust', 'mosaic', 'virus']):
            confidence = min(confidence + 10, 95)
        
        # Format the diagnosis
        formatted_diagnosis = f"{disease} ({crop_type.title()})"
        
        # Format treatment with all information
        formatted_treatment = f"""**Disease:** {disease}
**Severity:** {severity}
**Symptoms:** {symptoms}

**Treatment Recommendations:**
{treatment}

**Prevention Measures:**
{prevention}"""
        
        return formatted_diagnosis, confidence, formatted_treatment
        
    except Exception as e:
        print(f"[WARNING] Error parsing OpenAI response: {e}")
        # Fallback to original response
        return response_text, 75, response_text

def assess_image_quality(filepath):
    """Assess image quality and return quality score"""
    try:
        img = Image.open(filepath)
        width, height = img.size
        
        # Basic quality checks
        quality_score = 50  # Base score
        
        # Check image size
        if width >= 800 and height >= 600:
            quality_score += 20
        elif width >= 400 and height >= 300:
            quality_score += 10
        
        # Check if image is RGB (color)
        if img.mode == 'RGB':
            quality_score += 10
        
        # Check file size (larger files often have better quality)
        file_size = os.path.getsize(filepath)
        if file_size > 500000:  # > 500KB
            quality_score += 10
        elif file_size > 100000:  # > 100KB
            quality_score += 5
        
        return min(quality_score, 100)
    except Exception as e:
        print(f"[WARNING] Error assessing image quality: {e}")
        return 50

def reset_daily_usage_if_needed(user):
    """Reset daily usage counter if it's a new day"""
    today = datetime.datetime.utcnow().date()
    
    # Handle case where last_usage_reset is None (for existing users)
    if user.last_usage_reset is None:
        user.last_usage_reset = datetime.datetime.utcnow()
        user.openai_usage_today = 0
        session.commit()
        print(f"[INFO] Initialized usage tracking for user {user.username}")
        return
    
    last_reset = user.last_usage_reset.date()
    
    if today > last_reset:
        user.openai_usage_today = 0
        user.last_usage_reset = datetime.datetime.utcnow()
        session.commit()
        print(f"[INFO] Reset daily usage for user {user.username}")

def can_use_openai(user):
    """Check if user can use OpenAI service"""
    reset_daily_usage_if_needed(user)
    
    # Free users get 5 OpenAI calls per day
    # Premium users get unlimited calls
    if user.subscription_type == 'premium':
        return True
    else:
        return user.openai_usage_today < 5

def increment_openai_usage(user):
    """Increment OpenAI usage counter"""
    user.openai_usage_today += 1
    session.commit()
    print(f"[INFO] OpenAI usage incremented for user {user.username}. Today's usage: {user.openai_usage_today}")

def log_usage(user, service_type, crop_type=None, confidence_score=None):
    """Log usage for analytics"""
    usage_log = UsageLog(
        user_id=user.id,
        service_type=service_type,
        crop_type=crop_type,
        confidence_score=confidence_score,
        date=datetime.datetime.utcnow()
    )
    session.add(usage_log)
    session.commit()

def get_usage_stats(user):
    """Get user's usage statistics"""
    reset_daily_usage_if_needed(user)
    
    # Get today's usage
    today = datetime.datetime.utcnow().date()
    today_logs = session.query(UsageLog).filter(
        UsageLog.user_id == user.id,
        UsageLog.date >= today
    ).all()
    
    # Count by service type
    stats = {
        'openai_today': user.openai_usage_today,
        'local_model_today': len([log for log in today_logs if log.service_type == 'local_model']),
        'fallback_today': len([log for log in today_logs if log.service_type == 'fallback']),
        'total_today': len(today_logs),
        'subscription_type': user.subscription_type,
        'openai_limit': 5 if user.subscription_type == 'free' else 'unlimited'
    }
    
    return stats

def generate_crop_specific_fallback(crop_type):
    crop_recommendations = {
        'tomato': """Tomato Plant Health Recommendations:

Common Issues to Watch For:
• Blight (Early/Late)
• Bacterial spot
• Blossom end rot
• Aphids and whiteflies

General Care:
1. Watering: Water at soil level, avoid wetting leaves
2. Support: Use stakes or cages for proper support
3. Pruning: Remove suckers and lower leaves
4. Fertilizer: Use balanced fertilizer with calcium
5. Spacing: Ensure good air circulation between plants

Prevention:
• Rotate crops annually
• Use disease-resistant varieties
• Mulch around plants
• Monitor for pests regularly""",

        'potato': """**Potato Plant Health Recommendations:**

**Common Issues to Watch For:**
- Late blight
- Early blight
- Potato scab
- Colorado potato beetle

**General Care:**
1. **Planting**: Use certified seed potatoes
2. **Hilling**: Hill soil around plants as they grow
3. **Watering**: Consistent moisture, avoid overwatering
4. **Harvesting**: Harvest when foliage dies back
5. **Storage**: Store in cool, dark, dry place

**Prevention:**
- Rotate crops (don't plant in same spot for 3+ years)
- Remove volunteer potatoes
- Use disease-free seed
- Monitor for pests""",

        'corn': """**Corn/Maize Plant Health Recommendations:**

**Common Issues to Watch For:**
- Corn smut
- Rust
- Corn earworm
- Aphids

**General Care:**
1. **Planting**: Plant in blocks for better pollination
2. **Fertilizer**: High nitrogen fertilizer
3. **Watering**: Consistent moisture during tasseling
4. **Harvesting**: Harvest when kernels are milky
5. **Spacing**: Proper spacing for air circulation

**Prevention:**
- Rotate crops annually
- Use resistant varieties
- Monitor for pests
- Remove diseased plants immediately""",

        'wheat': """**Wheat Plant Health Recommendations:**

**Common Issues to Watch For:**
- Rust diseases
- Powdery mildew
- Fusarium head blight
- Aphids

**General Care:**
1. **Planting**: Use certified seed
2. **Fertilizer**: Balanced NPK fertilizer
3. **Watering**: Adequate moisture during growth
4. **Harvesting**: Harvest when grain is hard
5. **Storage**: Store in dry conditions

**Prevention:**
- Rotate crops
- Use resistant varieties
- Monitor for diseases
- Proper field sanitation""",

        'rice': """**Rice Plant Health Recommendations:**

**Common Issues to Watch For:**
- Rice blast
- Bacterial leaf blight
- Brown spot
- Stem borer

**General Care:**
1. **Water Management**: Maintain proper water levels
2. **Fertilizer**: Balanced fertilizer application
3. **Planting**: Proper spacing and depth
4. **Harvesting**: Harvest at optimal maturity
5. **Storage**: Proper drying and storage

**Prevention:**
- Use disease-resistant varieties
- Proper water management
- Field sanitation
- Monitor for pests and diseases"""
    }
    
    # Get specific recommendations or general ones
    if crop_type in crop_recommendations:
        return crop_recommendations[crop_type]
    else:
        return f"""**{crop_type.title()} Plant Health Recommendations:**

**General Care Guidelines:**
1. **Watering**: Provide consistent, appropriate moisture
2. **Fertilizer**: Use balanced fertilizer as needed
3. **Pruning**: Remove dead or diseased parts
4. **Spacing**: Ensure adequate air circulation
5. **Monitoring**: Regular inspection for pests/diseases

**Prevention Measures:**
- Use disease-resistant varieties when available
- Practice crop rotation
- Maintain good field hygiene
- Monitor weather conditions
- Consult local agricultural extension for specific advice

**Note**: This is general guidance. For specific {crop_type} issues, consult with a local agricultural expert."""

@login_manager.user_loader
def load_user(user_id):
    return session.query(User).get(int(user_id))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        
        user = session.query(User).filter_by(email=email).first()
        
        if user and user.check_password(password):
            login_user(user)
            next_page = request.args.get('next')
            return redirect(next_page or url_for('home'))
        else:
            flash('Invalid email or password')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        role = request.form['role']
        
        if password != confirm_password:
            flash('Passwords do not match')
            return render_template('register.html')
        
        if session.query(User).filter_by(email=email).first():
            flash('Email already registered')
            return render_template('register.html')
        
        if session.query(User).filter_by(username=username).first():
            flash('Username already taken')
            return render_template('register.html')
        
        user = User(username=username, email=email, role=role)
        user.set_password(password)
        
        session.add(user)
        session.commit()
        
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    crop = None
    crop_info = None
    soil_analysis = None
    
    if request.method == 'POST':
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temp = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        crop = recommend_crop(N, P, K, temp, humidity, ph, rainfall)
        
        # Generate detailed crop information
        if crop:
            crop_info = get_crop_details(crop)
            soil_analysis = analyze_soil_conditions(N, P, K, ph)

    return render_template('recommend.html', crop=crop, crop_info=crop_info, soil_analysis=soil_analysis)

def get_crop_details(crop_name):
    """Get detailed information about the recommended crop"""
    crop_details = {
        'rice': {
            'icon': '🌾',
            'season': 'Wet season (June-September)',
            'duration': '120-150 days',
            'water_needs': 'High (flooded fields)',
            'temperature_range': '20-35°C',
            'soil_type': 'Clay loam, alluvial',
            'market_value': 'High demand, stable prices',
            'disease_resistance': 'Moderate',
            'tips': [
                'Requires consistent water supply',
                'Plant in rows for better management',
                'Use certified seeds for better yield',
                'Monitor for blast disease',
                'Harvest when 80% grains are mature'
            ]
        },
        'maize': {
            'icon': '🌽',
            'season': 'Kharif (June-July)',
            'duration': '90-120 days',
            'water_needs': 'Moderate',
            'temperature_range': '18-30°C',
            'soil_type': 'Well-drained loamy soil',
            'market_value': 'Good demand, fluctuating prices',
            'disease_resistance': 'Good',
            'tips': [
                'Plant in rows 60cm apart',
                'Apply nitrogen in split doses',
                'Control weeds early',
                'Monitor for stem borer',
                'Harvest when kernels are hard'
            ]
        },
        'chickpea': {
            'icon': '🫘',
            'season': 'Rabi (October-November)',
            'duration': '120-150 days',
            'water_needs': 'Low to moderate',
            'temperature_range': '15-25°C',
            'soil_type': 'Well-drained sandy loam',
            'market_value': 'High protein demand',
            'disease_resistance': 'Good',
            'tips': [
                'Sow in rows 30cm apart',
                'Avoid waterlogging',
                'Use rhizobium inoculation',
                'Control pod borer',
                'Harvest when pods turn brown'
            ]
        },
        'kidneybeans': {
            'icon': '🫘',
            'season': 'Kharif (June-July)',
            'duration': '90-120 days',
            'water_needs': 'Moderate',
            'temperature_range': '20-30°C',
            'soil_type': 'Well-drained loamy soil',
            'market_value': 'High protein demand',
            'disease_resistance': 'Moderate',
            'tips': [
                'Plant in rows 45cm apart',
                'Provide support for climbing varieties',
                'Control aphids and beetles',
                'Harvest when pods are mature',
                'Store in cool, dry place'
            ]
        },
        'pigeonpeas': {
            'icon': '🫘',
            'season': 'Kharif (June-July)',
            'duration': '150-180 days',
            'water_needs': 'Low to moderate',
            'temperature_range': '20-35°C',
            'soil_type': 'Well-drained sandy loam',
            'market_value': 'Stable demand',
            'disease_resistance': 'Good',
            'tips': [
                'Sow in rows 60cm apart',
                'Drought tolerant crop',
                'Fix nitrogen in soil',
                'Control pod fly',
                'Harvest in multiple pickings'
            ]
        },
        'mothbeans': {
            'icon': '🫘',
            'season': 'Kharif (June-July)',
            'duration': '90-120 days',
            'water_needs': 'Low',
            'temperature_range': '25-35°C',
            'soil_type': 'Sandy loam',
            'market_value': 'Moderate demand',
            'disease_resistance': 'Good',
            'tips': [
                'Drought resistant crop',
                'Sow in rows 30cm apart',
                'Good for intercropping',
                'Control pod borer',
                'Harvest when pods are dry'
            ]
        },
        'mungbean': {
            'icon': '🫘',
            'season': 'Kharif (June-July)',
            'duration': '60-90 days',
            'water_needs': 'Low to moderate',
            'temperature_range': '25-35°C',
            'soil_type': 'Well-drained loamy soil',
            'market_value': 'High demand',
            'disease_resistance': 'Good',
            'tips': [
                'Short duration crop',
                'Sow in rows 30cm apart',
                'Good for crop rotation',
                'Control yellow mosaic virus',
                'Harvest when pods turn black'
            ]
        },
        'blackgram': {
            'icon': '🫘',
            'season': 'Kharif (June-July)',
            'duration': '90-120 days',
            'water_needs': 'Low to moderate',
            'temperature_range': '25-35°C',
            'soil_type': 'Well-drained loamy soil',
            'market_value': 'High protein demand',
            'disease_resistance': 'Good',
            'tips': [
                'Sow in rows 30cm apart',
                'Drought tolerant',
                'Fix nitrogen in soil',
                'Control pod borer',
                'Harvest when pods are mature'
            ]
        },
        'lentil': {
            'icon': '🫘',
            'season': 'Rabi (October-November)',
            'duration': '120-150 days',
            'water_needs': 'Low to moderate',
            'temperature_range': '15-25°C',
            'soil_type': 'Well-drained sandy loam',
            'market_value': 'High protein demand',
            'disease_resistance': 'Good',
            'tips': [
                'Sow in rows 30cm apart',
                'Avoid waterlogging',
                'Use rhizobium inoculation',
                'Control pod borer',
                'Harvest when pods turn brown'
            ]
        },
        'pomegranate': {
            'icon': '🍎',
            'season': 'Year-round',
            'duration': 'Perennial (3-4 years to fruit)',
            'water_needs': 'Moderate',
            'temperature_range': '15-35°C',
            'soil_type': 'Well-drained loamy soil',
            'market_value': 'High value crop',
            'disease_resistance': 'Good',
            'tips': [
                'Plant in pits 1m x 1m',
                'Prune regularly for shape',
                'Control fruit fly',
                'Harvest when fruits are mature',
                'Store in cool place'
            ]
        },
        'banana': {
            'icon': '🍌',
            'season': 'Year-round',
            'duration': 'Perennial (12-15 months)',
            'water_needs': 'High',
            'temperature_range': '20-35°C',
            'soil_type': 'Well-drained loamy soil',
            'market_value': 'High demand',
            'disease_resistance': 'Moderate',
            'tips': [
                'Plant suckers 2m apart',
                'Provide support during fruiting',
                'Control bunchy top virus',
                'Harvest when fingers are plump',
                'Remove old plants after harvest'
            ]
        },
        'mango': {
            'icon': '🥭',
            'season': 'Year-round',
            'duration': 'Perennial (3-5 years to fruit)',
            'water_needs': 'Moderate',
            'temperature_range': '20-35°C',
            'soil_type': 'Well-drained loamy soil',
            'market_value': 'High value crop',
            'disease_resistance': 'Moderate',
            'tips': [
                'Plant in pits 1m x 1m',
                'Prune for better fruiting',
                'Control anthracnose',
                'Harvest when fruits are mature',
                'Store in cool, dry place'
            ]
        },
        'grapes': {
            'icon': '🍇',
            'season': 'Year-round',
            'duration': 'Perennial (2-3 years to fruit)',
            'water_needs': 'Moderate',
            'temperature_range': '15-35°C',
            'soil_type': 'Well-drained loamy soil',
            'market_value': 'High value crop',
            'disease_resistance': 'Moderate',
            'tips': [
                'Plant in rows 3m apart',
                'Provide trellis support',
                'Prune regularly',
                'Control powdery mildew',
                'Harvest when berries are sweet'
            ]
        },
        'watermelon': {
            'icon': '🍉',
            'season': 'Summer (March-May)',
            'duration': '90-120 days',
            'water_needs': 'High',
            'temperature_range': '25-35°C',
            'soil_type': 'Well-drained sandy loam',
            'market_value': 'High demand in summer',
            'disease_resistance': 'Moderate',
            'tips': [
                'Plant in hills 2m apart',
                'Provide adequate water',
                'Control fruit fly',
                'Harvest when tendril dries',
                'Store in cool place'
            ]
        },
        'muskmelon': {
            'icon': '🍈',
            'season': 'Summer (March-May)',
            'duration': '90-120 days',
            'water_needs': 'Moderate',
            'temperature_range': '25-35°C',
            'soil_type': 'Well-drained sandy loam',
            'market_value': 'Good demand',
            'disease_resistance': 'Moderate',
            'tips': [
                'Plant in hills 1.5m apart',
                'Control powdery mildew',
                'Harvest when fruit slips easily',
                'Store in cool place',
                'Good for intercropping'
            ]
        },
        'apple': {
            'icon': '🍎',
            'season': 'Year-round',
            'duration': 'Perennial (3-4 years to fruit)',
            'water_needs': 'Moderate',
            'temperature_range': '10-25°C',
            'soil_type': 'Well-drained loamy soil',
            'market_value': 'High value crop',
            'disease_resistance': 'Moderate',
            'tips': [
                'Plant in pits 1m x 1m',
                'Prune for better fruiting',
                'Control apple scab',
                'Harvest when fruits are mature',
                'Store in cool, dry place'
            ]
        },
        'orange': {
            'icon': '🍊',
            'season': 'Year-round',
            'duration': 'Perennial (3-4 years to fruit)',
            'water_needs': 'Moderate',
            'temperature_range': '15-35°C',
            'soil_type': 'Well-drained loamy soil',
            'market_value': 'High demand',
            'disease_resistance': 'Good',
            'tips': [
                'Plant in pits 1m x 1m',
                'Prune regularly',
                'Control citrus canker',
                'Harvest when fruits are mature',
                'Store in cool place'
            ]
        },
        'papaya': {
            'icon': '🥭',
            'season': 'Year-round',
            'duration': 'Perennial (8-10 months to fruit)',
            'water_needs': 'Moderate',
            'temperature_range': '20-35°C',
            'soil_type': 'Well-drained loamy soil',
            'market_value': 'Good demand',
            'disease_resistance': 'Moderate',
            'tips': [
                'Plant in pits 1m x 1m',
                'Control papaya ring spot virus',
                'Harvest when fruits are mature',
                'Store in cool place',
                'Good for intercropping'
            ]
        },
        'coconut': {
            'icon': '🥥',
            'season': 'Year-round',
            'duration': 'Perennial (5-6 years to fruit)',
            'water_needs': 'High',
            'temperature_range': '20-35°C',
            'soil_type': 'Well-drained sandy loam',
            'market_value': 'High demand',
            'disease_resistance': 'Good',
            'tips': [
                'Plant in pits 1m x 1m',
                'Provide adequate water',
                'Control rhinoceros beetle',
                'Harvest when nuts are mature',
                'Good for coastal areas'
            ]
        },
        'cotton': {
            'icon': '🌾',
            'season': 'Kharif (June-July)',
            'duration': '150-180 days',
            'water_needs': 'Moderate',
            'temperature_range': '25-35°C',
            'soil_type': 'Well-drained loamy soil',
            'market_value': 'High demand',
            'disease_resistance': 'Moderate',
            'tips': [
                'Plant in rows 60cm apart',
                'Control bollworm',
                'Harvest when bolls are mature',
                'Store in dry place',
                'Good for crop rotation'
            ]
        },
        'jute': {
            'icon': '🌾',
            'season': 'Kharif (March-April)',
            'duration': '120-150 days',
            'water_needs': 'High',
            'temperature_range': '25-35°C',
            'soil_type': 'Alluvial soil',
            'market_value': 'Moderate demand',
            'disease_resistance': 'Good',
            'tips': [
                'Plant in rows 25cm apart',
                'Provide adequate water',
                'Control stem rot',
                'Harvest when plants are mature',
                'Ret in water for fiber extraction'
            ]
        },
        'coffee': {
            'icon': '☕',
            'season': 'Year-round',
            'duration': 'Perennial (3-4 years to fruit)',
            'water_needs': 'Moderate',
            'temperature_range': '15-25°C',
            'soil_type': 'Well-drained loamy soil',
            'market_value': 'High value crop',
            'disease_resistance': 'Moderate',
            'tips': [
                'Plant in pits 1m x 1m',
                'Provide shade',
                'Control coffee rust',
                'Harvest when berries are mature',
                'Process immediately after harvest'
            ]
        }
    }
    
    return crop_details.get(crop_name.lower(), {
        'icon': '🌱',
        'season': 'Varies by crop',
        'duration': 'Varies by crop',
        'water_needs': 'Moderate',
        'temperature_range': 'Varies by crop',
        'soil_type': 'Well-drained soil',
        'market_value': 'Good demand',
        'disease_resistance': 'Moderate',
        'tips': [
            'Follow recommended planting practices',
            'Monitor for pests and diseases',
            'Provide adequate water and nutrients',
            'Harvest at proper maturity',
            'Store properly after harvest'
        ]
    })

def analyze_soil_conditions(N, P, K, ph):
    """Analyze soil conditions and provide recommendations"""
    analysis = {
        'nitrogen_status': 'Optimal' if 50 <= N <= 150 else 'Low' if N < 50 else 'High',
        'phosphorus_status': 'Optimal' if 20 <= P <= 80 else 'Low' if P < 20 else 'High',
        'potassium_status': 'Optimal' if 30 <= K <= 120 else 'Low' if K < 30 else 'High',
        'ph_status': 'Optimal' if 6.0 <= ph <= 7.5 else 'Acidic' if ph < 6.0 else 'Alkaline',
        'overall_health': 'Good',
        'recommendations': []
    }
    
    # Generate recommendations based on soil analysis
    if analysis['nitrogen_status'] == 'Low':
        analysis['recommendations'].append('Add nitrogen-rich fertilizers like urea or compost')
    elif analysis['nitrogen_status'] == 'High':
        analysis['recommendations'].append('Reduce nitrogen application to prevent over-fertilization')
    
    if analysis['phosphorus_status'] == 'Low':
        analysis['recommendations'].append('Apply phosphorus-rich fertilizers like DAP or rock phosphate')
    elif analysis['phosphorus_status'] == 'High':
        analysis['recommendations'].append('Limit phosphorus application to prevent nutrient imbalance')
    
    if analysis['potassium_status'] == 'Low':
        analysis['recommendations'].append('Add potassium-rich fertilizers like MOP or wood ash')
    elif analysis['potassium_status'] == 'High':
        analysis['recommendations'].append('Reduce potassium application')
    
    if analysis['ph_status'] == 'Acidic':
        analysis['recommendations'].append('Apply lime to raise soil pH')
    elif analysis['ph_status'] == 'Alkaline':
        analysis['recommendations'].append('Apply sulfur or organic matter to lower soil pH')
    
    # Overall health assessment
    optimal_count = sum(1 for status in [analysis['nitrogen_status'], analysis['phosphorus_status'], 
                                        analysis['potassium_status']] if status == 'Optimal')
    if optimal_count >= 2 and analysis['ph_status'] == 'Optimal':
        analysis['overall_health'] = 'Excellent'
    elif optimal_count >= 1:
        analysis['overall_health'] = 'Good'
    else:
        analysis['overall_health'] = 'Needs Improvement'
    
    return analysis

@app.route('/weather', methods=['GET', 'POST'])
def weather():
    forecast = None
    pest_risk = None
    pest_tip = None
    five_day = []
    overall_risk = None

    if request.method == 'POST':
        city = request.form['city']
        api_key = "8dbcf2a7cc2a1d9db8b9144dce927eac"
        temperature, humidity, rainfall = get_weather_data(city, api_key)
        forecast = {
            "city": city,
            "temperature": temperature,
            "humidity": humidity,
            "rainfall": rainfall,
        }
        # Fetch 5-day forecast timeline
        try:
            five_day = get_weather(city) or []
            # Assess overall 5-day risk
            overall_risk = assess_5day_risk(five_day)
        except Exception as e:
            print(f"[ERROR] 5-day forecast fetch failed: {e}")
            five_day = []
            overall_risk = "Unable to assess 5-day risk"

        if temperature is not None:
            pest_risk = assess_pest_risk(temperature, humidity, rainfall)
        else:
            pest_risk = "Could not fetch weather data."

        # Enhanced prevention tips based on risk level
        prevention_tips = {
            "High risk of fungal disease and pest infestation": "🔴 CRITICAL: High risk detected today. Apply fungicides immediately, use insect traps, and monitor crops twice daily.",
            "Moderate risk of pest and disease issues": "🟡 CAUTION: Moderate risk present today. Monitor crops daily and consider preventive treatments.",
            "Moderate risk of fungal disease": "🟡 CAUTION: Fungal risk detected today. Apply preventive fungicides and improve air circulation.",
            "Low risk of pest or disease": "🟢 GOOD: No significant risk detected today. Maintain standard agricultural practices.",
            "Could not fetch weather data.": "❌ ERROR: Check your city name or try again later.",
            "High risk period - multiple high-risk days in forecast": "🔴 CRITICAL: Multiple high-risk days in the 5-day forecast. Take immediate preventive measures!",
            "Moderate risk period - some risk factors in forecast": "🟡 CAUTION: Some risk factors in the 5-day forecast. Monitor closely and prepare preventive measures.",
            "Low risk period - favorable forecast conditions": "🟢 GOOD: Favorable 5-day forecast conditions. Maintain standard practices."
        }
        pest_tip = prevention_tips.get(pest_risk, prevention_tips.get(overall_risk, "No specific tip available."))

    return render_template('weather.html', forecast=forecast, pest_risk=pest_risk, pest_tip=pest_tip, five_day=five_day, overall_risk=overall_risk)

@app.route('/predict-pest', methods=['GET', 'POST'])
def predict_pest():
    prediction = None
    if request.method == 'POST':
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        rainfall = float(request.form['rainfall'])
        crop_type = request.form['crop_type']

        # Enhanced pest prediction with multiple factors
        risk_level, confidence, detailed_analysis = predict_pest_risk_enhanced(
            temperature, humidity, rainfall, crop_type
        )

        # Generate comprehensive tips based on risk level and conditions
        tips = generate_comprehensive_tips(risk_level, temperature, humidity, rainfall, crop_type)

        return render_template('pest_form.html',
                               prediction=f"Risk Level: {risk_level.title()}",
                               crop=crop_type,
                               temp=temperature,
                               humidity=humidity,
                               rainfall=rainfall,
                               risk=risk_level,
                               tip=tips['primary'],
                               confidence=confidence,
                               detailed_analysis=detailed_analysis,
                               all_tips=tips)

    return render_template('pest_form.html', prediction=prediction)

def predict_pest_risk_enhanced(temperature, humidity, rainfall, crop_type):
    """
    Enhanced pest prediction using multiple factors and crop-specific knowledge
    """
    risk_score = 0
    factors = []
    
    # Temperature-based risk assessment
    if temperature < 10:
        risk_score += 0.1  # Very low pest activity
        factors.append("Low temperature reduces pest activity")
    elif 10 <= temperature < 20:
        risk_score += 0.3  # Moderate pest activity
        factors.append("Cool temperatures favor some pests")
    elif 20 <= temperature < 30:
        risk_score += 0.6  # High pest activity
        factors.append("Optimal temperature for most pests")
    elif 30 <= temperature < 35:
        risk_score += 0.8  # Very high pest activity
        factors.append("High temperature accelerates pest reproduction")
    else:  # > 35
        risk_score += 0.4  # Some pests die off
        factors.append("Extreme heat reduces some pest populations")
    
    # Humidity-based risk assessment
    if humidity < 30:
        risk_score += 0.2  # Low humidity reduces fungal diseases
        factors.append("Low humidity reduces fungal disease risk")
    elif 30 <= humidity < 50:
        risk_score += 0.4  # Moderate risk
        factors.append("Moderate humidity levels")
    elif 50 <= humidity < 70:
        risk_score += 0.7  # High risk for fungal diseases
        factors.append("High humidity favors fungal diseases")
    elif 70 <= humidity < 85:
        risk_score += 0.9  # Very high risk
        factors.append("Very high humidity creates ideal conditions for pests")
    else:  # > 85
        risk_score += 0.6  # Some pests struggle in extreme humidity
        factors.append("Extreme humidity can stress some pests")
    
    # Rainfall-based risk assessment
    if rainfall < 5:
        risk_score += 0.3  # Dry conditions reduce some pests
        factors.append("Low rainfall reduces water-dependent pests")
    elif 5 <= rainfall < 20:
        risk_score += 0.5  # Moderate risk
        factors.append("Moderate rainfall levels")
    elif 20 <= rainfall < 50:
        risk_score += 0.8  # High risk for water-related diseases
        factors.append("High rainfall increases disease risk")
    else:  # > 50
        risk_score += 0.4  # Flooding can kill some pests
        factors.append("Heavy rainfall can wash away some pests")
    
    # Crop-specific risk adjustments
    crop_risk_adjustments = {
        'rice': {'base_risk': 0.3, 'pests': ['rice blast', 'brown planthopper', 'stem borer']},
        'wheat': {'base_risk': 0.2, 'pests': ['rust', 'aphids', 'armyworm']},
        'maize': {'base_risk': 0.4, 'pests': ['corn borer', 'armyworm', 'rust']},
        'sugarcane': {'base_risk': 0.5, 'pests': ['red rot', 'smut', 'borer']},
        'cotton': {'base_risk': 0.6, 'pests': ['bollworm', 'aphids', 'whitefly']},
        'vegetables': {'base_risk': 0.7, 'pests': ['aphids', 'caterpillars', 'fungal diseases']}
    }
    
    crop_info = crop_risk_adjustments.get(crop_type, {'base_risk': 0.4, 'pests': ['general pests']})
    risk_score += crop_info['base_risk']
    factors.append(f"{crop_type.title()} is susceptible to: {', '.join(crop_info['pests'])}")
    
    # Seasonal adjustments (simplified - in real app, use actual dates)
    # High pest season (summer months)
    risk_score += 0.2
    factors.append("Current season favors pest activity")
    
    # Determine final risk level
    if risk_score >= 2.0:
        risk_level = 'high'
        confidence = min(95, int(risk_score * 20))
    elif risk_score >= 1.3:
        risk_level = 'medium'
        confidence = min(90, int(risk_score * 25))
    else:
        risk_level = 'low'
        confidence = min(85, int(risk_score * 30))
    
    detailed_analysis = {
        'risk_score': round(risk_score, 2),
        'factors': factors,
        'confidence': confidence,
        'crop_specific_pests': crop_info['pests']
    }
    
    return risk_level, confidence, detailed_analysis

def generate_comprehensive_tips(risk_level, temperature, humidity, rainfall, crop_type):
    """
    Generate comprehensive, actionable tips based on risk level and conditions
    """
    tips = {
        'primary': '',
        'immediate': [],
        'preventive': [],
        'monitoring': [],
        'environmental': []
    }
    
    if risk_level == 'high':
        tips['primary'] = "🚨 IMMEDIATE ACTION REQUIRED: High pest risk detected!"
        
        tips['immediate'] = [
            "Apply preventive fungicide within 24 hours",
            "Use neem oil spray immediately",
            "Set up pheromone traps around the field",
            "Increase monitoring frequency to twice daily"
        ]
        
        tips['preventive'] = [
            "Apply Trichoderma biofungicide weekly",
            "Use Bacillus thuringiensis for caterpillar control",
            "Implement crop rotation if possible",
            "Remove any infected plant material immediately"
        ]
        
        tips['monitoring'] = [
            "Check for pest eggs on underside of leaves",
            "Look for yellowing or wilting symptoms",
            "Monitor soil moisture levels",
            "Watch for unusual insect activity"
        ]
        
        tips['environmental'] = [
            "Improve air circulation around plants",
            "Reduce humidity through proper spacing",
            "Ensure good drainage to prevent waterlogging",
            "Consider shade management if temperature is too high"
        ]
        
    elif risk_level == 'medium':
        tips['primary'] = "⚠️ MODERATE RISK: Preventive measures recommended"
        
        tips['immediate'] = [
            "Apply neem oil spray within 48 hours",
            "Set up yellow sticky traps",
            "Monitor crops daily for early signs"
        ]
        
        tips['preventive'] = [
            "Apply mild repellents every 10 days",
            "Use beneficial insects like ladybugs",
            "Maintain proper plant spacing",
            "Apply compost tea for plant health"
        ]
        
        tips['monitoring'] = [
            "Check plants every 2-3 days",
            "Look for early pest damage",
            "Monitor weather changes",
            "Keep records of pest sightings"
        ]
        
        tips['environmental'] = [
            "Maintain optimal growing conditions",
            "Ensure proper irrigation",
            "Improve soil health with organic matter",
            "Consider mulching to regulate soil temperature"
        ]
        
    else:  # low risk
        tips['primary'] = "✅ LOW RISK: Maintain current practices"
        
        tips['immediate'] = [
            "Continue regular monitoring schedule",
            "Maintain current preventive measures"
        ]
        
        tips['preventive'] = [
            "Focus on soil health and nutrition",
            "Apply compost and organic fertilizers",
            "Practice crop rotation",
            "Use companion planting"
        ]
        
        tips['monitoring'] = [
            "Weekly crop inspection",
            "Monitor soil pH and nutrients",
            "Track weather patterns",
            "Document plant growth"
        ]
        
        tips['environmental'] = [
            "Maintain optimal growing conditions",
            "Ensure good soil drainage",
            "Provide adequate sunlight",
            "Keep field clean and weed-free"
        ]
    
    # Add condition-specific tips
    if humidity > 70:
        tips['environmental'].append("Reduce humidity through better ventilation")
    if temperature > 30:
        tips['environmental'].append("Provide shade during peak heat hours")
    if rainfall > 30:
        tips['environmental'].append("Improve drainage to prevent waterlogging")
    
    return tips

# New route for the marketplace with search functionality
@app.route('/marketplace')
def marketplace():
    search_query = request.args.get('search', '')
    category_filter = request.args.get('category', '')
    price_range = request.args.get('price_range', '')

    # Fetch all products from the database
    products_query = session.query(Product).join(User) # Join with User to access username

    # Apply search filter
    if search_query:
        products_query = products_query.filter(
            (Product.name.ilike(f'%{search_query}%')) |
            (Product.description.ilike(f'%{search_query}%')) |
            (User.username.ilike(f'%{search_query}%'))
        )

    # Apply category filter
    if category_filter:
        products_query = products_query.filter(Product.category.ilike(f'%{category_filter}%'))

    # Apply price range filter
    if price_range:
        if price_range == '0-50':
            products_query = products_query.filter(Product.price <= 50)
        elif price_range == '50-100':
            products_query = products_query.filter(Product.price > 50, Product.price <= 100)
        elif price_range == '100-200':
            products_query = products_query.filter(Product.price > 100, Product.price <= 200)
        elif price_range == '200+':
            products_query = products_query.filter(Product.price > 200)

    # Order by creation date (newest first)
    products_query = products_query.order_by(Product.created_at.desc())

    products = products_query.all()

    return render_template('marketplace.html', 
                         products=products, 
                         search_query=search_query,
                         category_filter=category_filter,
                         price_range=price_range)

# Product details route
@app.route('/product/<int:product_id>')
def product_details(product_id):
    product = session.query(Product).join(User).filter(Product.id == product_id).first()
    
    if not product:
        flash('Product not found.', 'error')
        return redirect(url_for('marketplace'))
    
    return render_template('product_details.html', product=product)

# New route for adding a product
@app.route('/add-product', methods=['GET', 'POST'])
@login_required
def add_product():
    if current_user.role != 'farmer':
        flash('Only farmers can add products')
        return redirect(url_for('marketplace'))
        
    if request.method == 'POST':
        name = request.form['name']
        description = request.form.get('description')
        price = float(request.form['price'])
        unit = request.form['unit']
        quantity_available = float(request.form['quantity_available'])
        category = request.form.get('category')
        
        # Handle image upload
        image_url = None
        if 'image' in request.files:
            file = request.files['image']
            if file and file.filename != '' and allowed_file(file.filename):
                # Generate unique filename
                filename = secure_filename(file.filename)
                # Add timestamp to make filename unique
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_')
                filename = timestamp + filename
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                # Store relative path for URL generation
                image_url = f"uploads/{filename}"

        if not name or not price or not unit or quantity_available is None:
            flash('Please fill in all required fields')
            return render_template('add_product.html')
        
        new_product = Product(
            name=name,
            farmer_id=current_user.id,
            description=description,
            price=price,
            unit=unit,
            quantity_available=quantity_available,
            image_url=image_url,
            category=category,
            created_at=datetime.datetime.utcnow(),
            updated_at=datetime.datetime.utcnow()
        )

        session.add(new_product)
        session.commit()
        
        flash('Product added successfully!')
        return redirect(url_for('marketplace'))

    return render_template('add_product.html')

@app.route('/dashboard')
@login_required
def dashboard():
    if current_user.role == 'farmer':
        # Get farmer's products
        products = session.query(Product).filter_by(farmer_id=current_user.id).all()
        
        # Calculate active products (products with quantity > 0)
        active_products = sum(1 for p in products if p.quantity_available > 0)
        
        # Get total sales (sum of all order items for this farmer's products)
        total_sales = 0
        pending_orders = 0
        for product in products:
            for order_item in product.order_items:
                total_sales += order_item.price_at_order * order_item.quantity
                if order_item.order.status == 'pending':
                    pending_orders += 1
        
        return render_template('dashboard.html',
                             products=products,
                             active_products=active_products,
                             total_sales=total_sales,
                             pending_orders=pending_orders)
    else:
        # Get consumer's orders
        orders = session.query(Order).filter_by(consumer_id=current_user.id).all()
        
        # Calculate statistics
        total_orders = len(orders)
        pending_orders = sum(1 for o in orders if o.status == 'pending')
        
        return render_template('dashboard.html',
                             orders=orders,
                             total_orders=total_orders,
                             pending_orders=pending_orders)

@app.route('/edit-product/<int:product_id>', methods=['GET', 'POST'])
@login_required
def edit_product(product_id):
    product = session.query(Product).get(product_id)

    if not product:
        flash('Product not found.', 'danger')
        return redirect(url_for('dashboard'))

    if product.farmer_id != current_user.id:
        flash('You do not have permission to edit this product.', 'danger')
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        # Update product details
        product.name = request.form['name']
        product.description = request.form.get('description')
        product.price = float(request.form['price'])
        product.unit = request.form['unit']
        product.quantity_available = float(request.form['quantity_available'])
        
        # Handle image upload
        if 'image' in request.files:
            file = request.files['image']
            if file and file.filename != '' and allowed_file(file.filename):
                # Generate unique filename
                filename = secure_filename(file.filename)
                # Add timestamp to make filename unique
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_')
                filename = timestamp + filename
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                # Store relative path for URL generation
                product.image_url = f"uploads/{filename}"
        product.category = request.form.get('category')
        product.updated_at = datetime.datetime.utcnow()

        session.commit()
        flash('Product updated successfully!', 'success')
        return redirect(url_for('dashboard'))

    return render_template('edit_product.html', product=product)

@app.route('/delete-product/<int:product_id>', methods=['POST'])
@login_required
def delete_product(product_id):
    product = session.query(Product).get(product_id)

    if not product:
        flash('Product not found.', 'danger')
        return redirect(url_for('dashboard'))

    if product.farmer_id != current_user.id:
        flash('You do not have permission to delete this product.', 'danger')
        return redirect(url_for('dashboard'))

    # Delete related order items first to avoid foreign key constraints
    session.query(OrderItem).filter_by(product_id=product.id).delete()
    session.delete(product)
    session.commit()

    flash('Product deleted successfully!', 'success')
    return redirect(url_for('dashboard'))

@app.route('/my-products')
@login_required
def farmer_products():
    if current_user.role != 'farmer':
        flash('You do not have access to this page.', 'danger')
        return redirect(url_for('dashboard'))

    products = session.query(Product).filter_by(farmer_id=current_user.id).all()

    return render_template('farmer_products.html', products=products)

@app.route('/add-to-cart/<int:product_id>', methods=['POST'])
@login_required
def add_to_cart(product_id):
    product = session.query(Product).get(product_id)

    if not product:
        flash('Product not found.', 'danger')
        return redirect(url_for('marketplace'))

    # Find the user's current active cart (Order with status 'cart')
    cart = session.query(Order).filter_by(consumer_id=current_user.id, status='cart').first()

    # If no cart exists, create one
    if not cart:
        cart = Order(
            consumer_id=current_user.id,
            total_price=0.0,
            status='cart',
            order_date=datetime.datetime.utcnow()
        )
        session.add(cart)
        session.commit() # Commit to get the cart ID

    # Check if the product is already in the cart
    order_item = session.query(OrderItem).filter_by(order_id=cart.id, product_id=product.id).first()

    if order_item:
        # If product is in cart, increment quantity
        order_item.quantity += 1
        order_item.price_at_order = product.price # Update price just in case it changed
    else:
        # If product is not in cart, create a new order item
        order_item = OrderItem(
            order_id=cart.id,
            product_id=product.id,
            quantity=1,
            price_at_order=product.price # Store price at the time of adding to cart
        )
        session.add(order_item)

    # Update total price of the cart (simple sum for now, can be more sophisticated)
    # Recalculate total price based on current items in cart
    cart.total_price = sum(item.quantity * item.price_at_order for item in cart.items)

    session.commit()

    flash(f'{product.name} added to your cart!', 'success')
    return redirect(url_for('marketplace'))

@app.route('/cart')
@login_required
def view_cart():
    if current_user.role != 'consumer':
        flash('Only consumers have a shopping cart.', 'danger')
        return redirect(url_for('home'))

    cart = session.query(Order).filter_by(consumer_id=current_user.id, status='cart').first()

    # If cart doesn't exist, create an empty one for the template
    if not cart:
        cart = Order(consumer_id=current_user.id, total_price=0.0, status='cart')

    return render_template('cart.html', cart=cart)

@app.route('/update-cart-item/<int:item_id>', methods=['POST'])
@login_required
def update_cart_item(item_id):
    if current_user.role != 'consumer':
        flash('Only consumers can update cart items.', 'danger')
        return redirect(url_for('home'))

    item = session.query(OrderItem).get(item_id)

    if not item:
        flash('Cart item not found.', 'danger')
        return redirect(url_for('view_cart'))

    # Ensure the item belongs to the current user's cart
    if item.order.consumer_id != current_user.id or item.order.status != 'cart':
        flash('You do not have permission to update this item.', 'danger')
        return redirect(url_for('view_cart'))

    try:
        new_quantity = int(request.form['quantity'])
        if new_quantity < 1:
            flash('Quantity must be at least 1.', 'warning')
            return redirect(url_for('view_cart'))

        item.quantity = new_quantity
        session.commit()

        # Recalculate cart total after updating item
        item.order.total_price = sum(cart_item.quantity * cart_item.price_at_order for cart_item in item.order.items)
        session.commit()

        flash('Quantity updated.', 'success')

    except ValueError:
        flash('Invalid quantity.', 'danger')

    return redirect(url_for('view_cart'))

@app.route('/remove-from-cart/<int:item_id>', methods=['POST'])
@login_required
def remove_from_cart(item_id):
    if current_user.role != 'consumer':
        flash('Only consumers can remove items from the cart.', 'danger')
        return redirect(url_for('home'))

    item = session.query(OrderItem).get(item_id)

    if not item:
        flash('Cart item not found.', 'danger')
        return redirect(url_for('view_cart'))

    # Ensure the item belongs to the current user's cart
    if item.order.consumer_id != current_user.id or item.order.status != 'cart':
        flash('You do not have permission to remove this item.', 'danger')
        return redirect(url_for('view_cart'))

    session.delete(item)
    session.commit()

    # Recalculate cart total after removing item
    if item.order.items:
        item.order.total_price = sum(cart_item.quantity * cart_item.price_at_order for cart_item in item.order.items)
    else:
        # If cart is empty after removal, set total to 0
        item.order.total_price = 0.0

    session.commit()

    flash('Item removed from cart.', 'success')
    return redirect(url_for('view_cart'))

@app.route('/checkout')
@login_required
def checkout():
    if current_user.role != 'consumer':
        flash('Only consumers can checkout.', 'danger')
        return redirect(url_for('home'))

    cart = session.query(Order).filter_by(consumer_id=current_user.id, status='cart').first()

    if not cart or not cart.items:
        flash('Your cart is empty.', 'warning')
        return redirect(url_for('marketplace'))

    return render_template('checkout.html', cart=cart)

@app.route('/openai-diagnosis', methods=['GET', 'POST'])
@login_required
def openai_diagnosis():
    usage_stats = get_usage_stats(current_user)
    
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No image uploaded')
            return redirect(request.url)
        
        file = request.files['image']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
            
        crop_type = request.form.get('crop_type', 'unknown')
        if not crop_type:
            flash('Please select crop type for accurate diagnosis')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            with open(filepath, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
            
            diagnosis = None
            confidence = 0
            treatment = None
            diagnosis_method = 'fallback'
            
            can_use_openai_service = OPENAI_AVAILABLE and can_use_openai(current_user)
            
            if can_use_openai_service:
                try:
                    service = get_openai_service()
                    if service:
                        result = service.analyze_crop_disease(base64_image, crop_type, current_user.id)
                        if result['success']:
                            raw_diagnosis = result['response']
                            diagnosis, confidence, treatment = parse_openai_response(raw_diagnosis, crop_type)
                            diagnosis_method = 'openai'
                            usage_logger.log_openai_usage(result['usage_data'])
                            increment_openai_usage(current_user)
                            log_usage(current_user, 'openai', crop_type, confidence)
                        else:
                            print(f"[WARNING] OpenAI API failed: {result['error']}")
                    else:
                        print("[WARNING] OpenAI service not available - API key not set")
                except Exception as e:
                    print(f"[WARNING] OpenAI service error: {e}")
            
            # ✅ SAFE FALLBACK BLOCK
            if not diagnosis:
                try:
                    if disease_model is not None and disease_class_names:
                        img = Image.open(filepath).convert('RGB')
                        img = img.resize((224, 224))
                        img_array = np.array(img) / 255.0
                        img_array = np.expand_dims(img_array, axis=0)
                        
                        predictions = disease_model.predict(img_array)
                        predicted_class_idx = np.argmax(predictions[0])
                        confidence_score = float(predictions[0][predicted_class_idx])
                        
                        if predicted_class_idx < len(disease_class_names):
                            disease_name = disease_class_names[predicted_class_idx]
                            diagnosis = f"Detected: {disease_name}"
                            confidence = int(confidence_score * 100)
                            diagnosis_method = 'local_model'
                            treatment = generate_treatment_recommendations(disease_name)
                            log_usage(current_user, 'local_model', crop_type, confidence)
                            usage_logger.log_local_model_usage(current_user.id, 'disease_model', confidence)
                        else:
                            raise Exception("Invalid prediction index")
                    else:
                        # 👇 This line prevents crash when model file missing
                        raise Exception("Disease model not found or not loaded")
                        
                except Exception as e:
                    print(f"[INFO] Falling back: {e}")
                    diagnosis = f"{crop_type.title()} Plant Health Analysis"
                    confidence = 70
                    treatment = generate_crop_specific_fallback(crop_type)
                    diagnosis_method = 'fallback'
                    log_usage(current_user, 'fallback', crop_type, confidence)
                    usage_logger.log_fallback_usage(current_user.id, 'crop_specific')
            
            # ✅ Save diagnosis history
            if current_user.is_authenticated:
                new_diagnosis = Diagnosis(
                    user_id=current_user.id,
                    image_path=filename,
                    disease_name=diagnosis.split(':')[1].strip() if ':' in diagnosis else diagnosis,
                    confidence=confidence / 100.0,
                    treatment=treatment,
                    diagnosis_method=diagnosis_method,
                    crop_type=crop_type,
                    date=datetime.datetime.utcnow()
                )
                session.add(new_diagnosis)
                session.commit()
            
            return render_template(
                'diagnosis_result.html',
                disease=diagnosis,
                image_path=filename,
                confidence=confidence,
                treatment=treatment,
                diagnosis_method=diagnosis_method,
                usage_stats=get_usage_stats(current_user)
            )

    return render_template('openai_diagnosis.html', usage_stats=usage_stats)


@app.route('/subscription')
@login_required
def subscription():
    """Subscription management page"""
    usage_stats = get_usage_stats(current_user)
    return render_template('subscription.html', usage_stats=usage_stats)

@app.route('/upgrade-premium')
@login_required
def upgrade_premium():
    """Upgrade user to premium subscription"""
    current_user.subscription_type = 'premium'
    current_user.subscription_expires = datetime.datetime.utcnow() + datetime.timedelta(days=30)  # 30-day trial
    session.commit()
    
    flash('Congratulations! You have been upgraded to Premium! Enjoy unlimited OpenAI diagnoses.', 'success')
    return redirect(url_for('subscription'))

@app.route('/diagnosis-history')
@login_required
def diagnosis_history():
    diagnoses = session.query(Diagnosis).filter_by(user_id=current_user.id).order_by(Diagnosis.date.desc()).all()
    return render_template('diagnosis_history.html', diagnoses=diagnoses)

@app.route('/forecast', methods=['GET', 'POST'])
@login_required
def forecast_yield():
    forecast_result = None
    if request.method == 'POST':
        # Get data from form
        crop_type = request.form['crop_type']
        area_planted = float(request.form['area_planted'])
        seed_rate = float(request.form['seed_rate'])
        expected_germination = float(request.form['expected_germination'])
        soil_quality = float(request.form['soil_quality'])
        
        # Enhanced yield prediction with multiple factors
        forecast_result = calculate_enhanced_yield_prediction(
            crop_type, area_planted, seed_rate, expected_germination, soil_quality
        )

        return render_template('forecast_result.html', forecast=forecast_result)

    return render_template('forecast.html')

def calculate_enhanced_yield_prediction(crop_type, area_planted, seed_rate, expected_germination, soil_quality):
    """
    Enhanced yield prediction using multiple factors and crop-specific knowledge
    """
    # Base yield per acre for different crops (kg/acre)
    base_yield_per_acre = {
        'wheat': {'base': 3000, 'optimal_seed_rate': 15, 'optimal_germination': 90},
        'rice': {'base': 4000, 'optimal_seed_rate': 20, 'optimal_germination': 85},
        'maize': {'base': 3500, 'optimal_seed_rate': 12, 'optimal_germination': 88},
        'sugarcane': {'base': 25000, 'optimal_seed_rate': 8, 'optimal_germination': 80},
        'cotton': {'base': 800, 'optimal_seed_rate': 5, 'optimal_germination': 85}
    }
    
    crop_info = base_yield_per_acre.get(crop_type.lower(), {'base': 2000, 'optimal_seed_rate': 10, 'optimal_germination': 85})
    base_yield = crop_info['base']
    
    # Calculate adjustment factors
    germination_factor = expected_germination / 100
    soil_factor = soil_quality / 10
    
    # Seed rate optimization factor
    optimal_seed_rate = crop_info['optimal_seed_rate']
    seed_rate_factor = 1.0
    if seed_rate < optimal_seed_rate * 0.8:
        seed_rate_factor = 0.7  # Under-seeding reduces yield
    elif seed_rate > optimal_seed_rate * 1.2:
        seed_rate_factor = 0.9  # Over-seeding can reduce yield due to competition
    
    # Germination optimization factor
    optimal_germination = crop_info['optimal_germination']
    germination_optimization_factor = 1.0
    if expected_germination < optimal_germination * 0.9:
        germination_optimization_factor = 0.8
    elif expected_germination > optimal_germination:
        germination_optimization_factor = 1.1
    
    # Area efficiency factor (larger areas may have slightly lower efficiency)
    area_factor = 1.0
    if area_planted > 50:
        area_factor = 0.95
    elif area_planted < 1:
        area_factor = 0.9
    
    # Calculate predicted yield with all factors
    predicted_yield = (base_yield * area_planted * germination_factor * soil_factor * 
                      seed_rate_factor * germination_optimization_factor * area_factor)
    
    # Calculate seed production (8% of yield can be used as seed)
    seed_production_factor = 0.08
    predicted_seed_production = predicted_yield * seed_production_factor
    
    # Calculate yield per acre
    yield_per_acre = predicted_yield / area_planted if area_planted > 0 else 0
    
    # Determine yield category
    yield_category = 'excellent' if yield_per_acre > base_yield * 0.9 else 'good' if yield_per_acre > base_yield * 0.7 else 'moderate'
    
    # Calculate confidence score
    confidence_factors = [
        germination_factor,
        soil_factor,
        seed_rate_factor,
        germination_optimization_factor,
        area_factor
    ]
    confidence_score = min(95, int(sum(confidence_factors) / len(confidence_factors) * 100))
    
    # Generate insights
    insights = generate_yield_insights(crop_type, yield_per_acre, base_yield, soil_quality, 
                                     expected_germination, seed_rate, optimal_seed_rate)
    
    forecast_result = {
        'crop_type': crop_type,
        'area_planted': area_planted,
        'predicted_yield': predicted_yield,
        'predicted_seed_production': predicted_seed_production,
        'yield_per_acre': yield_per_acre,
        'yield_category': yield_category,
        'confidence_score': confidence_score,
        'unit': 'kg',
        'seed_rate': seed_rate,
        'expected_germination': expected_germination,
        'soil_quality': soil_quality,
        'optimal_seed_rate': optimal_seed_rate,
        'insights': insights
    }
    
    return forecast_result

def generate_yield_insights(crop_type, yield_per_acre, base_yield, soil_quality, 
                          expected_germination, seed_rate, optimal_seed_rate):
    """
    Generate personalized insights based on the yield prediction
    """
    insights = []
    
    # Yield performance insight
    yield_percentage = (yield_per_acre / base_yield) * 100
    if yield_percentage > 90:
        insights.append(f"Excellent yield potential! Your predicted yield is {yield_percentage:.0f}% of the optimal base yield.")
    elif yield_percentage > 70:
        insights.append(f"Good yield potential. Your predicted yield is {yield_percentage:.0f}% of the optimal base yield.")
    else:
        insights.append(f"Moderate yield potential. Consider optimizing farming practices to improve yield.")
    
    # Soil quality insight
    if soil_quality >= 8:
        insights.append("Excellent soil quality! This will significantly contribute to your yield.")
    elif soil_quality >= 6:
        insights.append("Good soil quality. Consider soil testing for specific nutrient recommendations.")
    else:
        insights.append("Consider soil improvement measures like organic matter addition and proper fertilization.")
    
    # Germination insight
    if expected_germination >= 90:
        insights.append("High germination rate expected. This indicates good seed quality and planting conditions.")
    elif expected_germination >= 80:
        insights.append("Good germination rate. Ensure proper seed treatment and planting depth.")
    else:
        insights.append("Consider seed treatment and optimal planting conditions to improve germination.")
    
    # Seed rate insight
    if abs(seed_rate - optimal_seed_rate) <= optimal_seed_rate * 0.1:
        insights.append("Optimal seed rate! This will help achieve maximum yield potential.")
    elif seed_rate < optimal_seed_rate * 0.8:
        insights.append("Consider increasing seed rate to improve plant density and yield.")
    else:
        insights.append("Seed rate is higher than optimal. Consider reducing to avoid plant competition.")
    
    # Crop-specific insights
    crop_specific_insights = {
        'wheat': "Ensure proper irrigation during flowering stage for optimal grain development.",
        'rice': "Maintain consistent water levels and consider proper spacing for better tillering.",
        'maize': "Monitor for corn borer and ensure adequate pollination during flowering.",
        'sugarcane': "Regular irrigation and proper ratoon management will maximize yield.",
        'cotton': "Monitor for bollworm and ensure proper spacing for better boll development."
    }
    
    if crop_type.lower() in crop_specific_insights:
        insights.append(crop_specific_insights[crop_type.lower()])
    
    return insights

# New route for preventive measures
@app.route('/preventive_measures')
@login_required
def preventive_measures():
    # TODO: Add logic here to fetch/display preventive measures
    # You might want to pass parameters to this route to filter measures
    # based on predicted pest risk, crop type, etc.
    return render_template('preventive_measures.html')

# Health Monitoring Dashboard Routes
@app.route('/health-dashboard')
@login_required
def health_dashboard():
    """Main health monitoring dashboard with trend analysis"""
    # Get user's health monitoring data
    weather_data = session.query(WeatherData).filter_by(user_id=current_user.id).order_by(WeatherData.date.desc()).limit(30).all()
    pest_data = session.query(PestRiskData).filter_by(user_id=current_user.id).order_by(PestRiskData.date.desc()).limit(30).all()
    disease_data = session.query(DiseaseData).filter_by(user_id=current_user.id).order_by(DiseaseData.date.desc()).limit(30).all()
    
    # Calculate trends and correlations
    trends = calculate_health_trends(weather_data, pest_data, disease_data)
    
    return render_template('health_dashboard.html', 
                         weather_data=weather_data,
                         pest_data=pest_data,
                         disease_data=disease_data,
                         trends=trends)

@app.route('/add-health-data', methods=['GET', 'POST'])
@login_required
def add_health_data():
    """Add new health monitoring data"""
    if request.method == 'POST':
        data_type = request.form['data_type']
        
        if data_type == 'weather':
            try:
                weather = WeatherData(
                    user_id=current_user.id,
                    location=request.form['location'],
                    temperature=float(request.form['temperature']),
                    humidity=float(request.form['humidity']),
                    rainfall=float(request.form['rainfall']),
                    wind_speed=float(request.form.get('wind_speed', 0) or 0),
                    pressure=float(request.form.get('pressure', 0) or 0)
                )
                session.add(weather)
                print(f"[DEBUG] Weather data added: {weather.location}, {weather.temperature}°C")
            except Exception as e:
                print(f"[ERROR] Weather data save failed: {e}")
                flash(f'Error saving weather data: {str(e)}', 'error')
                return redirect(url_for('add_health_data'))
            
        elif data_type == 'pest_risk':
            try:
                pest_risk = PestRiskData(
                    user_id=current_user.id,
                    crop_type=request.form['crop_type'],
                    pest_risk_level=request.form['risk_level'],
                    risk_score=float(request.form['risk_score']),
                    temperature=float(request.form['pest_temperature']),
                    humidity=float(request.form['pest_humidity']),
                    rainfall=float(request.form['pest_rainfall'])
                )
                session.add(pest_risk)
                print(f"[DEBUG] Pest risk data added: {pest_risk.crop_type}, {pest_risk.pest_risk_level}")
            except Exception as e:
                print(f"[ERROR] Pest risk data save failed: {e}")
                flash(f'Error saving pest risk data: {str(e)}', 'error')
                return redirect(url_for('add_health_data'))
            
        elif data_type == 'disease':
            try:
                disease = DiseaseData(
                    user_id=current_user.id,
                    crop_type=request.form['crop_type'],
                    disease_name=request.form['disease_name'],
                    severity=request.form['severity'],
                    humidity=float(request.form['disease_humidity']),
                    temperature=float(request.form['disease_temperature']),
                    rainfall=float(request.form['disease_rainfall'])
                )
                session.add(disease)
                print(f"[DEBUG] Disease data added: {disease.disease_name}, {disease.severity}")
            except Exception as e:
                print(f"[ERROR] Disease data save failed: {e}")
                flash(f'Error saving disease data: {str(e)}', 'error')
                return redirect(url_for('add_health_data'))
        
        session.commit()
        print(f"[DEBUG] Data committed to database successfully")
        flash('✅ Health data saved successfully! Your data has been recorded.', 'success')
        print(f"[DEBUG] Redirecting to health dashboard")
        return redirect(url_for('health_dashboard'))
    
    return render_template('add_health_data.html')

@app.route('/api/health-trends')
@login_required
def api_health_trends():
    """API endpoint for health trend data (for AJAX requests)"""
    days = int(request.args.get('days', 30))
    
    # Get data for the specified number of days
    weather_data = session.query(WeatherData).filter_by(user_id=current_user.id).order_by(WeatherData.date.desc()).limit(days).all()
    pest_data = session.query(PestRiskData).filter_by(user_id=current_user.id).order_by(PestRiskData.date.desc()).limit(days).all()
    disease_data = session.query(DiseaseData).filter_by(user_id=current_user.id).order_by(DiseaseData.date.desc()).limit(days).all()
    
    # Format data for charts
    chart_data = format_health_data_for_charts(weather_data, pest_data, disease_data)
    
    return json.dumps(chart_data)

@app.route('/clear-health-data', methods=['POST'])
@login_required
def clear_health_data():
    """Clear all health monitoring data for the current user"""
    try:
        # Delete all health data for the current user
        session.query(WeatherData).filter_by(user_id=current_user.id).delete()
        session.query(PestRiskData).filter_by(user_id=current_user.id).delete()
        session.query(DiseaseData).filter_by(user_id=current_user.id).delete()
        
        session.commit()
        print(f"[DEBUG] Cleared all health data for user: {current_user.username}")
        flash('✅ All health data cleared successfully!', 'success')
        return redirect(url_for('health_dashboard'))
    except Exception as e:
        print(f"[ERROR] Failed to clear health data: {e}")
        flash(f'Error clearing data: {str(e)}', 'error')
        return redirect(url_for('health_dashboard'))

@app.route('/usage')
@login_required
def usage_dashboard():
    """
    Display API usage statistics and logs for the current user.
    Shows the last 20 entries in a Bootstrap-styled table.
    """
    try:
        # Get user's usage statistics
        user_stats = usage_logger.get_user_usage_stats(current_user.id, days=30)
        
        # Get recent usage logs (last 20 entries)
        recent_logs = session.query(ApiUsageLog).filter(
            ApiUsageLog.user_id == current_user.id
        ).order_by(ApiUsageLog.timestamp.desc()).limit(20).all()
        
        # Calculate additional statistics
        total_cost_this_month = sum(log.estimated_cost for log in recent_logs)
        total_tokens_this_month = sum(log.total_tokens for log in recent_logs)
        
        # Get model pricing information
        service = get_openai_service()
        model_pricing = service.get_model_pricing() if service else {}
        
        return render_template('usage_dashboard.html',
                             user_stats=user_stats,
                             recent_logs=recent_logs,
                             total_cost_this_month=round(total_cost_this_month, 6),
                             total_tokens_this_month=total_tokens_this_month,
                             model_pricing=model_pricing)
                             
    except Exception as e:
        print(f"[ERROR] Failed to load usage dashboard: {e}")
        flash(f'Error loading usage data: {str(e)}', 'error')
        return redirect(url_for('dashboard'))

@app.route('/api/usage-stats')
@login_required
def api_usage_stats():
    """
    API endpoint to return usage statistics in JSON format.
    Useful for AJAX requests and API integrations.
    """
    try:
        # Get user's usage statistics
        user_stats = usage_logger.get_user_usage_stats(current_user.id, days=30)
        
        # Convert datetime objects to strings for JSON serialization
        recent_usage_data = []
        for log in user_stats['recent_usage']:
            recent_usage_data.append({
                'id': log.id,
                'timestamp': log.timestamp.isoformat(),
                'model_name': log.model_name,
                'prompt_tokens': log.prompt_tokens,
                'completion_tokens': log.completion_tokens,
                'total_tokens': log.total_tokens,
                'estimated_cost': log.estimated_cost,
                'service_type': log.service_type,
                'error_message': log.error_message,
                'confidence_score': log.confidence_score
            })
        
        user_stats['recent_usage'] = recent_usage_data
        
        return json.dumps(user_stats)
        
    except Exception as e:
        print(f"[ERROR] Failed to get usage stats API: {e}")
        return json.dumps({'error': str(e)}), 500

@app.route('/admin/usage-overview')
@login_required
def admin_usage_overview():
    """
    Admin-only route to view aggregate usage statistics across all users.
    Only accessible by admin users.
    """
    # Check if user is admin (you might want to add an admin role to your User model)
    if current_user.role != 'admin':
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('dashboard'))
    
    try:
        # Get aggregate usage statistics
        aggregate_stats = usage_logger.get_all_usage_stats(days=30)
        
        return render_template('admin_usage_overview.html',
                             aggregate_stats=aggregate_stats)
                             
    except Exception as e:
        print(f"[ERROR] Failed to load admin usage overview: {e}")
        flash(f'Error loading usage overview: {str(e)}', 'error')
        return redirect(url_for('dashboard'))

def calculate_health_trends(weather_data, pest_data, disease_data):
    """Calculate health trends and correlations"""
    trends = {
        'rainfall_pest_correlation': 0,
        'humidity_disease_correlation': 0,
        'temperature_trend': 'stable',
        'risk_alerts': []
    }
    
    if len(weather_data) >= 2 and len(pest_data) >= 2:
        # Calculate correlation between rainfall and pest risk
        rainfall_values = [w.rainfall for w in weather_data[-10:]]
        pest_scores = [p.risk_score for p in pest_data[-10:]]
        
        if len(rainfall_values) == len(pest_scores):
            correlation = np.corrcoef(rainfall_values, pest_scores)[0, 1]
            trends['rainfall_pest_correlation'] = round(correlation, 3)
    
    if len(weather_data) >= 2 and len(disease_data) >= 2:
        # Calculate correlation between humidity and disease severity
        humidity_values = [w.humidity for w in weather_data[-10:]]
        disease_severity = [1 if d.severity == 'high' else 0.5 if d.severity == 'medium' else 0 for d in disease_data[-10:]]
        
        if len(humidity_values) == len(disease_severity):
            correlation = np.corrcoef(humidity_values, disease_severity)[0, 1]
            trends['humidity_disease_correlation'] = round(correlation, 3)
    
    # Generate risk alerts
    if pest_data:
        recent_pest = pest_data[0]
        if recent_pest.risk_score > 0.7:
            trends['risk_alerts'].append(f"High pest risk detected for {recent_pest.crop_type}")
    
    if disease_data:
        recent_disease = disease_data[0]
        if recent_disease.severity == 'high':
            trends['risk_alerts'].append(f"High disease severity: {recent_disease.disease_name}")
    
    return trends

def format_health_data_for_charts(weather_data, pest_data, disease_data):
    """Format health data for Chart.js visualization"""
    chart_data = {
        'labels': [],
        'rainfall': [],
        'humidity': [],
        'temperature': [],
        'pest_risk': [],
        'disease_severity': []
    }
    
    # Get all unique dates and sort them
    all_dates = set()
    for w in weather_data:
        all_dates.add(w.date.date())
    for p in pest_data:
        all_dates.add(p.date.date())
    for d in disease_data:
        all_dates.add(d.date.date())
    
    sorted_dates = sorted(list(all_dates))[-30:]  # Last 30 days
    
    # Create date lookup dictionaries
    weather_by_date = {w.date.date(): w for w in weather_data}
    pest_by_date = {p.date.date(): p for p in pest_data}
    disease_by_date = {d.date.date(): d for d in disease_data}
    
    # Format for charts - one entry per date
    for date in sorted_dates:
        date_str = date.strftime('%Y-%m-%d')
        chart_data['labels'].append(date_str)
        
        # Get data for this date
        weather = weather_by_date.get(date)
        pest = pest_by_date.get(date)
        disease = disease_by_date.get(date)
        
        # Add data (0 if no data for this date)
        chart_data['rainfall'].append(weather.rainfall if weather else 0)
        chart_data['humidity'].append(weather.humidity if weather else 0)
        chart_data['temperature'].append(weather.temperature if weather else 0)
        chart_data['pest_risk'].append(pest.risk_score if pest else 0)
        
        if disease:
            severity_score = 1 if disease.severity == 'high' else 0.5 if disease.severity == 'medium' else 0
            chart_data['disease_severity'].append(severity_score)
        else:
            chart_data['disease_severity'].append(0)
    
    # Add individual data points for scatter plots
    chart_data['individual_weather'] = []
    chart_data['individual_pest'] = []
    chart_data['individual_disease'] = []
    
    # Add all individual weather records
    for w in weather_data:
        chart_data['individual_weather'].append({
            'humidity': w.humidity,
            'temperature': w.temperature,
            'rainfall': w.rainfall,
            'date': w.date.strftime('%Y-%m-%d')
        })
    
    # Add all individual pest records
    for p in pest_data:
        chart_data['individual_pest'].append({
            'risk_score': p.risk_score,
            'crop_type': p.crop_type,
            'risk_level': p.pest_risk_level,
            'date': p.date.strftime('%Y-%m-%d')
        })
    
    # Add all individual disease records
    for d in disease_data:
        chart_data['individual_disease'].append({
            'severity': d.severity,
            'disease_name': d.disease_name,
            'humidity': d.humidity,
            'temperature': d.temperature,
            'date': d.date.strftime('%Y-%m-%d')
        })
    
    print(f"[DEBUG] Chart data generated: {len(chart_data['labels'])} dates")
    print(f"[DEBUG] Individual records: {len(chart_data['individual_weather'])} weather, {len(chart_data['individual_pest'])} pest, {len(chart_data['individual_disease'])} disease")
    
    return chart_data

if __name__ == '__main__':
    # In a real application, database creation and potentially data seeding
    # would be handled separately, e.g., with Flask-Migrate.
    # For this example, we assume models.py has been run once to create the db.
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)

    
