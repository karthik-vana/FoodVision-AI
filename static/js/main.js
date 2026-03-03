/**
 * Food Image Classification — Main JavaScript
 * Handles: Drag & Drop, Image Preview, Theme Toggle, Toasts, Spinner, Animations
 * Author: Karthik Vana
 */

// ═══════════════════════════════════════════════════════════════════════
// THEME MANAGER
// ═══════════════════════════════════════════════════════════════════════

class ThemeManager {
    constructor() {
        this.theme = localStorage.getItem('theme') || 'light';
        this.toggleBtn = document.getElementById('themeToggle');
        this.init();
    }

    init() {
        document.documentElement.setAttribute('data-theme', this.theme);
        this.updateIcon();

        if (this.toggleBtn) {
            this.toggleBtn.addEventListener('click', () => this.toggle());
        }
    }

    toggle() {
        this.theme = this.theme === 'light' ? 'dark' : 'light';
        document.documentElement.setAttribute('data-theme', this.theme);
        localStorage.setItem('theme', this.theme);
        this.updateIcon();
    }

    updateIcon() {
        if (!this.toggleBtn) return;
        const icon = this.toggleBtn.querySelector('i');
        if (icon) {
            icon.className = this.theme === 'dark'
                ? 'bi bi-sun-fill'
                : 'bi bi-moon-fill';
        }
        const text = this.toggleBtn.querySelector('.theme-text');
        if (text) {
            text.textContent = this.theme === 'dark' ? 'Light' : 'Dark';
        }
    }
}


// ═══════════════════════════════════════════════════════════════════════
// IMAGE UPLOAD HANDLER (Drag & Drop)
// ═══════════════════════════════════════════════════════════════════════

class ImageUploadHandler {
    constructor() {
        this.uploadZone = document.getElementById('uploadZone');
        this.fileInput = document.getElementById('imageInput');
        this.previewContainer = document.getElementById('imagePreview');
        this.previewImage = document.getElementById('previewImg');
        this.fileName = document.getElementById('fileName');
        this.predictBtn = document.getElementById('predictBtn');

        if (this.uploadZone && this.fileInput) {
            this.init();
        }
    }

    init() {
        // Click to browse
        this.uploadZone.addEventListener('click', (e) => {
            if (e.target.closest('.browse-btn') || e.target === this.uploadZone || e.target.closest('.upload-icon')) {
                this.fileInput.click();
            }
        });

        // File selected via input
        this.fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleFile(e.target.files[0]);
            }
        });

        // Drag & Drop events
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            this.uploadZone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
            });
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            this.uploadZone.addEventListener(eventName, () => {
                this.uploadZone.classList.add('drag-over');
            });
        });

        ['dragleave', 'drop'].forEach(eventName => {
            this.uploadZone.addEventListener(eventName, () => {
                this.uploadZone.classList.remove('drag-over');
            });
        });

        this.uploadZone.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFile(files[0]);
            }
        });
    }

    handleFile(file) {
        // Validate file type
        const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/webp', 'image/bmp'];
        if (!allowedTypes.includes(file.type)) {
            ToastManager.show('Invalid file type. Please upload PNG, JPG, JPEG, WEBP, or BMP.', 'error');
            return;
        }

        // Validate file size (16 MB max)
        if (file.size > 16 * 1024 * 1024) {
            ToastManager.show('File too large. Maximum size is 16 MB.', 'error');
            return;
        }

        // Update file input
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        this.fileInput.files = dataTransfer.files;

        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            if (this.previewImage) {
                this.previewImage.src = e.target.result;
            }
            if (this.previewContainer) {
                this.previewContainer.style.display = 'block';
            }
            if (this.fileName) {
                this.fileName.textContent = file.name;
            }
        };
        reader.readAsDataURL(file);

        // Enable predict button
        if (this.predictBtn) {
            this.predictBtn.disabled = false;
        }

        ToastManager.show(`"${file.name}" selected successfully.`, 'success');
    }
}


// ═══════════════════════════════════════════════════════════════════════
// TOAST NOTIFICATION MANAGER
// ═══════════════════════════════════════════════════════════════════════

class ToastManager {
    static container = null;

    static getContainer() {
        if (!this.container) {
            this.container = document.createElement('div');
            this.container.className = 'toast-container';
            document.body.appendChild(this.container);
        }
        return this.container;
    }

    static show(message, type = 'info', duration = 4000) {
        const container = this.getContainer();

        const icons = {
            error: 'bi bi-exclamation-circle-fill',
            success: 'bi bi-check-circle-fill',
            info: 'bi bi-info-circle-fill',
        };

        const toast = document.createElement('div');
        toast.className = `toast-item toast-${type}`;
        toast.innerHTML = `
            <i class="toast-icon ${icons[type] || icons.info}"></i>
            <span class="toast-message">${message}</span>
            <button class="toast-close" onclick="this.parentElement.remove()">
                <i class="bi bi-x"></i>
            </button>
        `;

        container.appendChild(toast);

        // Auto remove
        setTimeout(() => {
            toast.classList.add('hiding');
            setTimeout(() => toast.remove(), 400);
        }, duration);
    }
}


// ═══════════════════════════════════════════════════════════════════════
// LOADING SPINNER
// ═══════════════════════════════════════════════════════════════════════

class SpinnerManager {
    static show() {
        const overlay = document.getElementById('spinnerOverlay');
        if (overlay) {
            overlay.classList.add('active');
        }
    }

    static hide() {
        const overlay = document.getElementById('spinnerOverlay');
        if (overlay) {
            overlay.classList.remove('active');
        }
    }
}


// ═══════════════════════════════════════════════════════════════════════
// FORM SUBMISSION HANDLER
// ═══════════════════════════════════════════════════════════════════════

class FormHandler {
    constructor() {
        this.form = document.getElementById('predictForm');
        if (this.form) {
            this.init();
        }
    }

    init() {
        this.form.addEventListener('submit', (e) => {
            const fileInput = document.getElementById('imageInput');
            if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
                e.preventDefault();
                ToastManager.show('Please upload an image first.', 'error');
                return;
            }

            // Show loading spinner
            SpinnerManager.show();

            // Disable submit button
            const btn = document.getElementById('predictBtn');
            if (btn) {
                btn.disabled = true;
                btn.innerHTML = '<i class="bi bi-hourglass-split"></i> Predicting...';
            }
        });
    }
}


// ═══════════════════════════════════════════════════════════════════════
// METRIC BAR ANIMATION (Result Page)
// ═══════════════════════════════════════════════════════════════════════

class MetricAnimator {
    constructor() {
        this.bars = document.querySelectorAll('.metric-bar-fill');
        if (this.bars.length > 0) {
            this.init();
        }
    }

    init() {
        // Use IntersectionObserver for scroll-triggered animation
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const bar = entry.target;
                    const targetWidth = bar.getAttribute('data-width');
                    setTimeout(() => {
                        bar.style.width = targetWidth + '%';
                    }, 200);
                    observer.unobserve(bar);
                }
            });
        }, { threshold: 0.3 });

        this.bars.forEach(bar => {
            bar.style.width = '0%';
            observer.observe(bar);
        });
    }
}


// ═══════════════════════════════════════════════════════════════════════
// LANDING PAGE PARTICLES
// ═══════════════════════════════════════════════════════════════════════

class ParticleGenerator {
    constructor() {
        this.container = document.querySelector('.particles');
        if (this.container) {
            this.generate(40);
        }
    }

    generate(count) {
        for (let i = 0; i < count; i++) {
            const particle = document.createElement('div');
            particle.className = 'particle';
            particle.style.left = Math.random() * 100 + '%';
            particle.style.animationDuration = (Math.random() * 10 + 8) + 's';
            particle.style.animationDelay = Math.random() * 8 + 's';
            particle.style.width = (Math.random() * 3 + 2) + 'px';
            particle.style.height = particle.style.width;
            this.container.appendChild(particle);
        }
    }
}


// ═══════════════════════════════════════════════════════════════════════
// FLASH MESSAGES (from Flask)
// ═══════════════════════════════════════════════════════════════════════

class FlashMessageHandler {
    constructor() {
        this.init();
    }

    init() {
        const flashMessages = document.querySelectorAll('.flash-data');
        flashMessages.forEach(el => {
            const message = el.getAttribute('data-message');
            const category = el.getAttribute('data-category') || 'info';
            if (message) {
                ToastManager.show(message, category);
            }
            el.remove();
        });
    }
}


// ═══════════════════════════════════════════════════════════════════════
// SCROLL REVEAL ANIMATIONS (Intersection Observer)
// ═══════════════════════════════════════════════════════════════════════

class ScrollRevealManager {
    constructor() {
        this.elements = document.querySelectorAll('.reveal-on-scroll');
        if (this.elements.length > 0) {
            this.init();
        }
    }

    init() {
        const observerOptions = {
            root: null,
            rootMargin: '0px 0px -50px 0px',
            threshold: 0.15
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('revealed');
                    // Don't unobserve — keep it revealed forever
                    observer.unobserve(entry.target);
                }
            });
        }, observerOptions);

        this.elements.forEach(el => observer.observe(el));
    }
}


// ═══════════════════════════════════════════════════════════════════════
// INITIALIZE ALL ON DOM READY
// ═══════════════════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
    // Core
    new ThemeManager();
    new FlashMessageHandler();

    // Page-specific
    new ImageUploadHandler();
    new FormHandler();
    new MetricAnimator();
    new ParticleGenerator();
    new ScrollRevealManager();
});
