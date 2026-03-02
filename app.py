"""
Food Image Classification Web Application
==========================================
Main Flask application with routes for landing, prediction, result, and about pages.
Uses OOP-based utility classes for model loading, image processing, prediction,
nutrition lookup, metrics reading, and Redis caching.

Author: Karthik Vana
"""

import os
import uuid
import logging
from flask import (
    Flask, render_template, request, redirect,
    url_for, flash, jsonify
)
from werkzeug.utils import secure_filename
from config import Config

# ─── Logging Configuration ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# APPLICATION FACTORY
# ═══════════════════════════════════════════════════════════════════════

def create_app(config_class=Config):
    """
    Flask Application Factory.
    
    Args:
        config_class: Configuration class to use.
    
    Returns:
        Flask: Configured Flask application instance.
    """
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Ensure upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # ─── Initialize Utility Objects ─────────────────────────────────
    from utils import (
        ModelLoader, ImagePreprocessor, Predictor,
        MetricsReader, NutritionLoader, RedisCache
    )

    try:
        model_loader = ModelLoader(app.config['MODEL_PATHS'])
    except Exception as e:
        logger.error("ModelLoader init failed: %s", e)
        model_loader = None

    image_preprocessor = ImagePreprocessor(app.config['IMAGE_SIZE'])

    predictor = Predictor(app.config['CLASS_LABELS'])

    try:
        metrics_reader = MetricsReader(app.config['METRICS_PATHS'])
    except Exception as e:
        logger.error("MetricsReader init failed: %s", e)
        metrics_reader = None

    try:
        nutrition_loader = NutritionLoader(app.config['NUTRITION_PATH'])
    except Exception as e:
        logger.error("NutritionLoader init failed: %s", e)
        nutrition_loader = None

    redis_cache = RedisCache(
        host=app.config['REDIS_HOST'],
        port=app.config['REDIS_PORT'],
        db=app.config['REDIS_DB'],
        password=app.config['REDIS_PASSWORD'],
        ttl=app.config['REDIS_CACHE_TTL'],
    )

    # ─── Helper Function ────────────────────────────────────────────
    def allowed_file(filename: str) -> bool:
        """Check if uploaded file has an allowed extension."""
        return (
            '.' in filename
            and filename.rsplit('.', 1)[1].lower()
            in app.config['ALLOWED_EXTENSIONS']
        )

    # ═══════════════════════════════════════════════════════════════
    # ROUTES
    # ═══════════════════════════════════════════════════════════════

    @app.route('/')
    def landing():
        """Landing page with animated intro."""
        return render_template('landing.html')

    @app.route('/home')
    def home():
        """Main prediction page with upload and model selection."""
        class_labels = app.config['CLASS_LABELS']
        models = app.config['MODEL_DISPLAY_NAMES']
        redis_status = redis_cache.get_status()
        return render_template(
            'index.html',
            class_labels=class_labels,
            models=models,
            redis_status=redis_status,
        )

    @app.route('/predict', methods=['POST'])
    def predict():
        """
        Handle image upload and run prediction.
        Validates input, checks cache, runs model inference,
        fetches nutrition and metrics, then renders result.
        """
        try:
            # ── Validate File Upload ────────────────────────────────
            if 'image' not in request.files:
                flash('No image file uploaded.', 'error')
                return redirect(url_for('home'))

            file = request.files['image']
            if file.filename == '':
                flash('No file selected.', 'error')
                return redirect(url_for('home'))

            if not allowed_file(file.filename):
                flash(
                    'Invalid file type. Allowed: PNG, JPG, JPEG, WEBP, BMP.',
                    'error'
                )
                return redirect(url_for('home'))

            # ── Validate Model Selection ────────────────────────────
            model_key = request.form.get('model', 'custom_cnn')
            if model_key not in app.config['MODEL_PATHS']:
                flash('Invalid model selection.', 'error')
                return redirect(url_for('home'))

            # ── Read Image Bytes for Caching ─────────────────────────
            image_bytes = file.read()
            file.seek(0)  # Reset for saving

            # ── Check Redis Cache ────────────────────────────────────
            cached_result = redis_cache.get(image_bytes, model_key)
            if cached_result:
                logger.info("Serving cached result.")
                return render_template(
                    'result.html',
                    result=cached_result,
                    from_cache=True,
                )

            # ── Save Uploaded File ──────────────────────────────────
            filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            logger.info("Image saved: %s", filepath)

            # ── Load Model ──────────────────────────────────────────
            if model_loader is None:
                flash('Model loader not initialized.', 'error')
                return redirect(url_for('home'))

            model = model_loader.load_model(model_key)

            # ── Preprocess Image ────────────────────────────────────
            target_size_override = None
            if hasattr(model, 'input_shape') and len(model.input_shape) >= 3:
                # Check for standard (None, H, W, 3) shape
                target_size_override = (model.input_shape[1], model.input_shape[2])
                
            img_tensor = image_preprocessor.preprocess(filepath, target_size=target_size_override)

            # ── Run Prediction ──────────────────────────────────────
            prediction = predictor.predict(model, img_tensor)

            # ── Fetch Nutrition Data ─────────────────────────────────
            nutrition = {}
            if nutrition_loader:
                nutrition = nutrition_loader.get_nutrition(
                    prediction['predicted_class']
                )

            # ── Fetch Model Metrics ──────────────────────────────────
            metrics = {}
            if metrics_reader:
                metrics = metrics_reader.read_metrics(model_key)

            # ── Build Result ─────────────────────────────────────────
            result = {
                'predicted_class': prediction['predicted_class'],
                'confidence': prediction['confidence'],
                'top_predictions': prediction['all_probabilities'],
                'nutrition': nutrition,
                'metrics': metrics,
                'model_used': app.config['MODEL_DISPLAY_NAMES'].get(
                    model_key, model_key
                ),
                'image_url': url_for(
                    'static', filename=f'uploads/{filename}'
                ),
            }

            # ── Cache Result ─────────────────────────────────────────
            redis_cache.set(image_bytes, model_key, result)

            # ── Cleanup (optional: delete after prediction) ───────────
            # os.remove(filepath)

            return render_template(
                'result.html',
                result=result,
                from_cache=False,
            )

        except Exception as e:
            logger.error("Prediction error: %s", str(e), exc_info=True)
            flash(f'Prediction failed: {str(e)}', 'error')
            return redirect(url_for('home'))

    @app.route('/about')
    def about():
        """About page with project details."""
        return render_template('about.html')

    @app.route('/api/status')
    def api_status():
        """Health check / status endpoint."""
        return jsonify({
            'status': 'ok',
            'redis': redis_cache.get_status(),
            'models_available': list(app.config['MODEL_PATHS'].keys()),
            'total_classes': len(app.config['CLASS_LABELS']),
        })

    # ─── Error Handlers ─────────────────────────────────────────────
    @app.errorhandler(404)
    def not_found(error):
        flash('Page not found.', 'error')
        return redirect(url_for('landing'))

    @app.errorhandler(413)
    def file_too_large(error):
        flash('File too large. Maximum size is 16 MB.', 'error')
        return redirect(url_for('home'))

    @app.errorhandler(500)
    def internal_error(error):
        flash('An internal server error occurred.', 'error')
        return redirect(url_for('home'))

    return app


# ═══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    application = create_app()
    application.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
    )
