// AYM Website Interactive Features
document.addEventListener('DOMContentLoaded', function() {
    
    // Mobile Navigation Toggle
    const navToggle = document.getElementById('nav-toggle');
    const navMenu = document.getElementById('nav-menu');
    const navLinks = document.querySelectorAll('.nav-link');
    
    if (navToggle && navMenu) {
        navToggle.addEventListener('click', function() {
            navMenu.classList.toggle('active');
            navToggle.classList.toggle('active');
        });
        
        // Close mobile menu when clicking on a link
        navLinks.forEach(link => {
            link.addEventListener('click', function() {
                navMenu.classList.remove('active');
                navToggle.classList.remove('active');
            });
        });
    }
    
    // Smooth Scrolling for Navigation Links
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            const targetSection = document.querySelector(targetId);
            
            if (targetSection) {
                const offsetTop = targetSection.offsetTop - 70; // Account for fixed navbar
                window.scrollTo({
                    top: offsetTop,
                    behavior: 'smooth'
                });
            }
        });
    });
    
    // Navbar Background Change on Scroll
    const navbar = document.getElementById('navbar');
    if (navbar) {
        window.addEventListener('scroll', function() {
            if (window.scrollY > 50) {
                navbar.style.backgroundColor = 'rgba(255, 255, 255, 0.95)';
                navbar.style.backdropFilter = 'blur(10px)';
            } else {
                navbar.style.backgroundColor = '#ffffff';
                navbar.style.backdropFilter = 'none';
            }
        });
    }
    
    // Active Navigation Link Highlighting
    const sections = document.querySelectorAll('section[id]');
    
    function highlightNavLink() {
        const scrollPos = window.scrollY + 100;
        
        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.offsetHeight;
            const sectionId = section.getAttribute('id');
            const navLink = document.querySelector(`.nav-link[href="#${sectionId}"]`);
            
            if (scrollPos >= sectionTop && scrollPos < sectionTop + sectionHeight) {
                navLinks.forEach(link => link.classList.remove('active'));
                if (navLink) {
                    navLink.classList.add('active');
                }
            }
        });
    }
    
    window.addEventListener('scroll', highlightNavLink);
    
    // Contact Form Handling
    const contactForm = document.getElementById('contactForm');
    if (contactForm) {
        contactForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Get form data
            const formData = new FormData(this);
            const name = formData.get('name');
            const email = formData.get('email');
            const subject = formData.get('subject');
            const message = formData.get('message');
            
            // Basic validation
            if (!name || !email || !subject || !message) {
                showNotification('Please fill in all fields.', 'error');
                return;
            }
            
            if (!isValidEmail(email)) {
                showNotification('Please enter a valid email address.', 'error');
                return;
            }
            
            // Simulate form submission
            showNotification('Thank you for your message! We will get back to you soon.', 'success');
            this.reset();
        });
    }
    
    // Email validation function
    function isValidEmail(email) {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return emailRegex.test(email);
    }
    
    // Notification system
    function showNotification(message, type = 'info') {
        // Remove existing notifications
        const existingNotifications = document.querySelectorAll('.notification');
        existingNotifications.forEach(notification => notification.remove());
        
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <span class="notification-message">${message}</span>
                <button class="notification-close">&times;</button>
            </div>
        `;
        
        // Add styles
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: ${type === 'success' ? '#2b8500' : type === 'error' ? '#dc3545' : '#007f98'};
            color: white;
            padding: 15px 20px;
            border-radius: 8px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
            z-index: 10000;
            max-width: 400px;
            animation: slideInRight 0.3s ease-out;
        `;
        
        // Add to page
        document.body.appendChild(notification);
        
        // Close button functionality
        const closeBtn = notification.querySelector('.notification-close');
        closeBtn.addEventListener('click', function() {
            notification.style.animation = 'slideOutRight 0.3s ease-out';
            setTimeout(() => notification.remove(), 300);
        });
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.style.animation = 'slideOutRight 0.3s ease-out';
                setTimeout(() => notification.remove(), 300);
            }
        }, 5000);
    }
    
    // Add CSS for notifications
    const notificationStyles = document.createElement('style');
    notificationStyles.textContent = `
        @keyframes slideInRight {
            from {
                transform: translateX(100%);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }
        
        @keyframes slideOutRight {
            from {
                transform: translateX(0);
                opacity: 1;
            }
            to {
                transform: translateX(100%);
                opacity: 0;
            }
        }
        
        .notification-content {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 15px;
        }
        
        .notification-close {
            background: none;
            border: none;
            color: white;
            font-size: 20px;
            cursor: pointer;
            padding: 0;
            line-height: 1;
        }
        
        .notification-close:hover {
            opacity: 0.8;
        }
    `;
    document.head.appendChild(notificationStyles);
    
    // Intersection Observer for Animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);
    
    // Observe elements for animation
    const animatedElements = document.querySelectorAll('.ministry-card, .event-card, .resource-card, .stat-item');
    animatedElements.forEach(element => {
        element.style.opacity = '0';
        element.style.transform = 'translateY(30px)';
        element.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(element);
    });
    
    // Counter Animation for Stats
    function animateCounter(element, target, duration = 2000) {
        let start = 0;
        const increment = target / (duration / 16);
        
        function updateCounter() {
            start += increment;
            if (start < target) {
                element.textContent = Math.floor(start).toLocaleString();
                requestAnimationFrame(updateCounter);
            } else {
                element.textContent = target.toLocaleString();
            }
        }
        
        updateCounter();
    }
    
    // Animate stats when they come into view
    const statNumbers = document.querySelectorAll('.stat-number');
    const statsObserver = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const text = entry.target.textContent;
                const number = parseInt(text.replace(/[^\d]/g, ''));
                if (number) {
                    animateCounter(entry.target, number);
                }
                statsObserver.unobserve(entry.target);
            }
        });
    }, { threshold: 0.5 });
    
    statNumbers.forEach(stat => statsObserver.observe(stat));
    
    // Lazy Loading for Images
    const images = document.querySelectorAll('img[data-src]');
    const imageObserver = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const img = entry.target;
                img.src = img.dataset.src;
                img.classList.remove('lazy');
                imageObserver.unobserve(img);
            }
        });
    });
    
    images.forEach(img => imageObserver.observe(img));
    
    // Back to Top Button
    const backToTopBtn = document.createElement('button');
    backToTopBtn.innerHTML = '<i class="fas fa-arrow-up"></i>';
    backToTopBtn.className = 'back-to-top';
    backToTopBtn.style.cssText = `
        position: fixed;
        bottom: 30px;
        right: 30px;
        width: 50px;
        height: 50px;
        background: linear-gradient(135deg, #2b8500, #4a9b2f);
        color: white;
        border: none;
        border-radius: 50%;
        cursor: pointer;
        font-size: 18px;
        box-shadow: 0 4px 20px rgba(43, 133, 0, 0.3);
        opacity: 0;
        visibility: hidden;
        transition: all 0.3s ease;
        z-index: 1000;
    `;
    
    document.body.appendChild(backToTopBtn);
    
    // Show/hide back to top button
    window.addEventListener('scroll', function() {
        if (window.scrollY > 300) {
            backToTopBtn.style.opacity = '1';
            backToTopBtn.style.visibility = 'visible';
        } else {
            backToTopBtn.style.opacity = '0';
            backToTopBtn.style.visibility = 'hidden';
        }
    });
    
    // Back to top functionality
    backToTopBtn.addEventListener('click', function() {
        window.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
    });
    
    // Hover effects for cards
    const cards = document.querySelectorAll('.ministry-card, .event-card, .resource-card');
    cards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-5px)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
        });
    });
    
    // Form validation enhancements
    const formInputs = document.querySelectorAll('input, textarea');
    formInputs.forEach(input => {
        input.addEventListener('blur', function() {
            validateField(this);
        });
        
        input.addEventListener('input', function() {
            clearFieldError(this);
        });
    });
    
    function validateField(field) {
        const value = field.value.trim();
        const fieldName = field.name;
        
        clearFieldError(field);
        
        if (!value) {
            showFieldError(field, `${getFieldLabel(fieldName)} is required.`);
            return false;
        }
        
        if (fieldName === 'email' && !isValidEmail(value)) {
            showFieldError(field, 'Please enter a valid email address.');
            return false;
        }
        
        return true;
    }
    
    function showFieldError(field, message) {
        field.style.borderColor = '#dc3545';
        
        let errorElement = field.parentNode.querySelector('.field-error');
        if (!errorElement) {
            errorElement = document.createElement('div');
            errorElement.className = 'field-error';
            errorElement.style.cssText = `
                color: #dc3545;
                font-size: 0.875rem;
                margin-top: 5px;
            `;
            field.parentNode.appendChild(errorElement);
        }
        errorElement.textContent = message;
    }
    
    function clearFieldError(field) {
        field.style.borderColor = '#ddd';
        const errorElement = field.parentNode.querySelector('.field-error');
        if (errorElement) {
            errorElement.remove();
        }
    }
    
    function getFieldLabel(fieldName) {
        const labels = {
            'name': 'Name',
            'email': 'Email',
            'subject': 'Subject',
            'message': 'Message'
        };
        return labels[fieldName] || fieldName;
    }
    
    // Performance optimization: Debounce scroll events
    function debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }
    
    // Apply debouncing to scroll events
    const debouncedScrollHandler = debounce(function() {
        highlightNavLink();
    }, 10);
    
    window.addEventListener('scroll', debouncedScrollHandler);
    
    // Carousel functionality
    let currentSlideIndex = 0;
    const slides = document.querySelectorAll('.carousel-slide');
    const indicators = document.querySelectorAll('.indicator');
    const totalSlides = slides.length;

    function showSlide(index) {
        // Remove active class from all slides and indicators
        slides.forEach(slide => slide.classList.remove('active'));
        indicators.forEach(indicator => indicator.classList.remove('active'));
        
        // Add active class to current slide and indicator
        if (slides[index]) {
            slides[index].classList.add('active');
        }
        if (indicators[index]) {
            indicators[index].classList.add('active');
        }
    }

    function changeSlide(direction) {
        currentSlideIndex += direction;
        
        if (currentSlideIndex >= totalSlides) {
            currentSlideIndex = 0;
        } else if (currentSlideIndex < 0) {
            currentSlideIndex = totalSlides - 1;
        }
        
        showSlide(currentSlideIndex);
    }

    function currentSlide(index) {
        currentSlideIndex = index - 1;
        showSlide(currentSlideIndex);
    }

    // Auto-advance carousel every 5 seconds
    setInterval(() => {
        changeSlide(1);
    }, 5000);

    // Bible verse rotation
    const bibleVerses = [
        {
            text: "Don't let anyone look down on you because you are young, but set an example for the believers in speech, in conduct, in love, in faith and in purity.",
            reference: "1 Timothy 4:12 (NIV)"
        },
        {
            text: "I can do all this through him who gives me strength.",
            reference: "Philippians 4:13 (NIV)"
        },
        {
            text: "For I know the plans I have for you, declares the Lord, plans to prosper you and not to harm you, to give you hope and a future.",
            reference: "Jeremiah 29:11 (NIV)"
        },
        {
            text: "Trust in the Lord with all your heart and lean not on your own understanding; in all your ways submit to him, and he will make your paths straight.",
            reference: "Proverbs 3:5-6 (NIV)"
        },
        {
            text: "And we know that in all things God works for the good of those who love him, who have been called according to his purpose.",
            reference: "Romans 8:28 (NIV)"
        },
        {
            text: "But seek first his kingdom and his righteousness, and all these things will be given to you as well.",
            reference: "Matthew 6:33 (NIV)"
        },
        {
            text: "The Lord your God is with you, the Mighty Warrior who saves. He will take great delight in you; in his love he will no longer rebuke you, but will rejoice over you with singing.",
            reference: "Zephaniah 3:17 (NIV)"
        },
        {
            text: "Therefore, if anyone is in Christ, the new creation has come: The old has gone, the new is here!",
            reference: "2 Corinthians 5:17 (NIV)"
        }
    ];

    function updateVerse() {
        const randomIndex = Math.floor(Math.random() * bibleVerses.length);
        const verse = bibleVerses[randomIndex];
        
        const verseElement = document.getElementById('current-verse');
        const referenceElement = document.getElementById('verse-reference');
        
        if (verseElement && referenceElement) {
            verseElement.textContent = `"${verse.text}"`;
            referenceElement.textContent = verse.reference;
        }
    }

    // Gallery Filter Functionality
    const filterButtons = document.querySelectorAll('.filter-btn');
    const albumSections = document.querySelectorAll('.album-section');
    
    filterButtons.forEach(button => {
        button.addEventListener('click', function() {
            const filter = this.getAttribute('data-filter');
            
            // Update active button
            filterButtons.forEach(btn => btn.classList.remove('active'));
            this.classList.add('active');
            
            // Filter albums
            albumSections.forEach(section => {
                const category = section.getAttribute('data-category');
                
                if (filter === 'all' || category === filter) {
                    section.classList.remove('hidden');
                } else {
                    section.classList.add('hidden');
                }
            });
        });
    });

    // Photo Lightbox Functionality
    const photoItems = document.querySelectorAll('.photo-item');
    
    photoItems.forEach(item => {
        item.addEventListener('click', function() {
            const img = this.querySelector('.photo-img');
            const title = this.querySelector('.photo-overlay h4').textContent;
            const description = this.querySelector('.photo-overlay p').textContent;
            
            showLightbox(img.src, title, description);
        });
    });
    
    function showLightbox(imageSrc, title, description) {
        // Create lightbox overlay
        const lightbox = document.createElement('div');
        lightbox.className = 'lightbox-overlay';
        lightbox.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.9);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 10000;
            cursor: pointer;
        `;
        
        // Create lightbox content
        const lightboxContent = document.createElement('div');
        lightboxContent.style.cssText = `
            max-width: 90%;
            max-height: 90%;
            position: relative;
            text-align: center;
        `;
        
        const lightboxImg = document.createElement('img');
        lightboxImg.src = imageSrc;
        lightboxImg.style.cssText = `
            max-width: 100%;
            max-height: 80vh;
            border-radius: 8px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
        `;
        
        const lightboxInfo = document.createElement('div');
        lightboxInfo.style.cssText = `
            color: white;
            margin-top: 20px;
        `;
        lightboxInfo.innerHTML = `
            <h3 style="margin-bottom: 10px; color: white;">${title}</h3>
            <p style="color: rgba(255, 255, 255, 0.8); margin: 0;">${description}</p>
        `;
        
        lightboxContent.appendChild(lightboxImg);
        lightboxContent.appendChild(lightboxInfo);
        lightbox.appendChild(lightboxContent);
        
        // Add close button
        const closeBtn = document.createElement('button');
        closeBtn.innerHTML = '&times;';
        closeBtn.style.cssText = `
            position: absolute;
            top: -40px;
            right: 0;
            background: none;
            border: none;
            color: white;
            font-size: 30px;
            cursor: pointer;
            padding: 5px 10px;
        `;
        lightboxContent.appendChild(closeBtn);
        
        // Add to page
        document.body.appendChild(lightbox);
        document.body.style.overflow = 'hidden';
        
        // Close functionality
        function closeLightbox() {
            document.body.removeChild(lightbox);
            document.body.style.overflow = 'auto';
        }
        
        lightbox.addEventListener('click', function(e) {
            if (e.target === lightbox) {
                closeLightbox();
            }
        });
        
        closeBtn.addEventListener('click', closeLightbox);
        
        // Close on escape key
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape') {
                closeLightbox();
            }
        });
    }

    // Spiritual Corner Functionality
    function updateCurrentDate() {
        const now = new Date();
        const options = { 
            weekday: 'long', 
            year: 'numeric', 
            month: 'long', 
            day: 'numeric' 
        };
        const dateElement = document.getElementById('current-date');
        if (dateElement) {
            dateElement.textContent = now.toLocaleDateString('en-US', options);
        }
    }

    // Daily Devotional Data
    const devotionalData = [
        {
            title: "Setting an Example",
            verse: "Don't let anyone look down on you because you are young, but set an example for the believers in speech, in conduct, in love, in faith and in purity.",
            reference: "1 Timothy 4:12 (NIV)",
            reflection: "As young people, we often feel that our age limits our ability to make a difference. But God's Word reminds us that youth is not a barrier to spiritual leadership. We can be examples of Christ's love and character in everything we do - in our words, actions, relationships, and faith. Today, let's commit to being positive examples in our families, schools, and communities.",
            prayer: "Lord, help me to be a positive example to others today. Give me the wisdom to speak words that build up, the strength to act with integrity, and the love to serve others selflessly. May my life reflect Your character and draw others closer to You. Amen."
        },
        {
            title: "Strength in Christ",
            verse: "I can do all this through him who gives me strength.",
            reference: "Philippians 4:13 (NIV)",
            reflection: "Life can be challenging, especially as a young person facing school, relationships, and decisions about the future. But we don't have to face these challenges alone. Christ promises to give us strength for whatever we encounter. When we rely on Him, we can overcome obstacles and accomplish great things for His kingdom.",
            prayer: "Dear Jesus, thank You for being my source of strength. Help me to remember that I can do all things through You. When I feel weak or overwhelmed, remind me to turn to You for the strength I need. Amen."
        },
        {
            title: "God's Plans for You",
            verse: "For I know the plans I have for you, declares the Lord, plans to prosper you and not to harm you, to give you hope and a future.",
            reference: "Jeremiah 29:11 (NIV)",
            reflection: "Sometimes we worry about our future - what career to choose, where to go to school, who to marry. But God has a plan for each of us, and His plans are always good. He wants to prosper us and give us hope. Trust in His timing and His wisdom as you make decisions about your future.",
            prayer: "Father, thank You for having a plan for my life. Help me to trust in Your timing and Your wisdom. Guide me in the decisions I need to make, and give me peace knowing that Your plans for me are good. Amen."
        },
        {
            title: "Trust in the Lord",
            verse: "Trust in the Lord with all your heart and lean not on your own understanding; in all your ways submit to him, and he will make your paths straight.",
            reference: "Proverbs 3:5-6 (NIV)",
            reflection: "It's easy to rely on our own understanding and try to figure everything out ourselves. But God's wisdom is far greater than ours. When we trust Him completely and submit our ways to Him, He promises to guide us and make our paths clear. This doesn't mean life will be easy, but it means we'll have divine guidance.",
            prayer: "Lord, help me to trust You with all my heart. When I don't understand what's happening in my life, remind me to lean on You rather than my own understanding. Guide my steps and make my path clear. Amen."
        }
    ];

    function loadNewDevotional() {
        const randomIndex = Math.floor(Math.random() * devotionalData.length);
        const devotional = devotionalData[randomIndex];
        
        document.getElementById('daily-verse-title').textContent = devotional.title;
        document.getElementById('daily-verse-text').textContent = `"${devotional.verse}"`;
        document.getElementById('daily-verse-reference').textContent = devotional.reference;
        document.getElementById('daily-reflection').textContent = devotional.reflection;
        document.getElementById('daily-prayer').textContent = devotional.prayer;
    }

    function shareDevotional() {
        const verse = document.getElementById('daily-verse-text').textContent;
        const reference = document.getElementById('daily-verse-reference').textContent;
        const title = document.getElementById('daily-verse-title').textContent;
        
        const shareText = `${title}\n\n${verse}\n- ${reference}\n\nShared from Adventist Youth Ministries`;
        
        if (navigator.share) {
            navigator.share({
                title: 'Daily Devotional',
                text: shareText,
                url: window.location.href
            });
        } else {
            // Fallback for browsers that don't support Web Share API
            navigator.clipboard.writeText(shareText).then(() => {
                showNotification('Devotional copied to clipboard!', 'success');
            });
        }
    }

    // Prayer Request Form
    const prayerRequestForm = document.getElementById('prayerRequestForm');
    if (prayerRequestForm) {
        prayerRequestForm.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const prayerText = this.querySelector('#prayer-request').value;
            const isAnonymous = this.querySelector('input[name="anonymous"]').checked;
            
            if (!prayerText.trim()) {
                showNotification('Please enter your prayer request.', 'error');
                return;
            }
            
            // Simulate prayer request submission
            showNotification('Your prayer request has been submitted. We will pray for you!', 'success');
            this.reset();
        });
    }

    // Initialize Spiritual Corner
    updateCurrentDate();
    
    // Make functions globally available
    window.changeSlide = changeSlide;
    window.currentSlide = currentSlide;
    window.updateVerse = updateVerse;
    window.loadNewDevotional = loadNewDevotional;
    window.shareDevotional = shareDevotional;

    // Initialize page
    console.log('AYM Website loaded successfully!');
});
