// Floating Navigation Bar
(function() {
    'use strict';

    // Navigation configurations for different sections
    const sectionConfigs = {
        'ml_fundamentals': {
            title: 'ML Fundamentals',
            sections: [
                { name: 'Overview', url: 'ml_fundamentals/ML_Fundamentals_Overview/' },
                { name: 'Feature Engineering', url: 'ml_fundamentals/feature_engineering/data_types_and_normalization/' },
                { name: 'Model Evaluation', url: 'ml_fundamentals/model_evaluation/evaluation_methods/' },
                { name: 'Regularization', url: 'ml_fundamentals/regularization/overfitting_underfitting/' },
                { name: 'Classical Algorithms', url: 'ml_fundamentals/classical_algorithms/linear_regression/' },
                { name: 'Unsupervised Learning', url: 'ml_fundamentals/unsupervised_learning/k_nearest_neighbors/' }
            ]
        },
        'probability_and_markov': {
            title: 'Probability & Markov',
            sections: [
                { name: 'Overview', url: 'probability_and_markov/Probability_and_Markov_Overview/' },
                { name: "Bayes' Rule", url: 'probability_and_markov/probability_and_markov_sections/conditional_probability_and_bayes_rule/' },
                { name: 'Naive Bayes', url: 'probability_and_markov/probability_and_markov_sections/naive_bayes_and_gaussian_naive_bayes/' },
                { name: 'Joint & Marginal', url: 'probability_and_markov/probability_and_markov_sections/joint_and_marginal_distributions/' }
            ]
        },
        'neural_networks': {
            title: 'Neural Networks',
            sections: [
                { name: 'Overview', url: 'neural_networks_and_deep_learning/Neural_Networks_and_Deep_Learning_Overview/' },
                { name: 'Perceptron Algorithm', url: 'neural_networks_and_deep_learning/neural_networks_sections/Introduction_to_Perceptron_Algorithm/' }
            ]
        },
        'language_model': {
            title: 'Language Models',
            sections: [
                { name: 'N-gram Language Modeling', url: 'language_model/Ngram_Language_Modeling/' }
            ]
        },
        'engineering': {
            title: 'Engineering & Data Structure',
            sections: [
                { name: 'Frequently Used', url: 'engineering_and_data_structure/Engineering_and_Data_Structure_Questions/' }
            ]
        }
    };

    // Create floating navigation element
    function createFloatingNav() {
        const nav = document.createElement('div');
        nav.className = 'floating-nav';
        nav.innerHTML = `
            <div class="floating-nav-header">
                <h3 id="floating-nav-title">Navigation</h3>
                <button class="floating-nav-toggle" id="floating-nav-toggle" aria-label="Toggle navigation">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M3 12h18M3 6h18M3 18h18"/>
                    </svg>
                </button>
            </div>
            <div class="floating-nav-content" id="floating-nav-content">
                <ul id="floating-nav-list"></ul>
            </div>
        `;
        document.body.appendChild(nav);
        return nav;
    }

    // Get current section from URL
    function getCurrentSection() {
        const path = window.location.pathname;
        console.log('Current path:', path); // Debug log
        
        // Check specific patterns for MkDocs URLs
        if (path.includes('/ml_fundamentals/') || path.includes('/ML_fundamentals/')) {
            console.log('Detected ML Fundamentals section'); // Debug log
            return 'ml_fundamentals';
        }
        if (path.includes('/probability_and_markov/') || path.includes('/Probability_and_Markov/')) {
            return 'probability_and_markov';
        }
        if (path.includes('/neural_networks_and_deep_learning/') || path.includes('/Neural_Networks/')) {
            return 'neural_networks';
        }
        if (path.includes('/language_model/') || path.includes('/Ngram_Language_Modeling/')) {
            return 'language_model';
        }
        if (path.includes('/engineering_and_data_structure/') || path.includes('/Engineering_and_Data_Structure/')) {
            return 'engineering';
        }
        
        console.log('No section detected'); // Debug log
        return null;
    }

    // Update floating navigation content
    function updateFloatingNav() {
        const nav = document.querySelector('.floating-nav');
        const titleEl = document.getElementById('floating-nav-title');
        const listEl = document.getElementById('floating-nav-list');
        
        if (!nav || !titleEl || !listEl) {
            console.log('Nav elements not found'); // Debug log
            return;
        }

        const currentSection = getCurrentSection();
        console.log('Current section:', currentSection); // Debug log
        
        if (!currentSection || !sectionConfigs[currentSection]) {
            nav.classList.remove('visible');
            return;
        }

        const config = sectionConfigs[currentSection];
        titleEl.textContent = config.title;
        
        // Clear existing list
        listEl.innerHTML = '';
        
        // Add navigation items
        config.sections.forEach(section => {
            const li = document.createElement('li');
            const a = document.createElement('a');
            a.href = section.url;
            a.textContent = section.name;
            
            // Check if this is the current page
            if (window.location.pathname.includes(section.url.replace('/', ''))) {
                a.classList.add('active');
            }
            
            li.appendChild(a);
            listEl.appendChild(li);
        });
        
        // Show the navigation (but collapsed by default)
        nav.classList.add('visible');
        nav.classList.remove('expanded');
        console.log('Floating nav should be visible now'); // Debug log
    }

    // Handle navigation clicks
    function handleNavClick(event) {
        if (event.target.tagName === 'A') {
            // Navigate to the page
            window.location.href = event.target.getAttribute('href');
        }
    }

    // Handle toggle button clicks
    function handleToggleClick() {
        const nav = document.querySelector('.floating-nav');
        if (nav) {
            nav.classList.toggle('expanded');
        }
    }

    // Initialize floating navigation
    function initFloatingNav() {
        console.log('Initializing floating nav'); // Debug log
        const nav = createFloatingNav();
        
        // Add click handlers
        nav.addEventListener('click', handleNavClick);
        
        const toggleBtn = document.getElementById('floating-nav-toggle');
        if (toggleBtn) {
            toggleBtn.addEventListener('click', handleToggleClick);
        }
        
        // Update on page load
        updateFloatingNav();
        
        // Update on navigation changes
        let currentPath = window.location.pathname;
        const observer = new MutationObserver(() => {
            if (window.location.pathname !== currentPath) {
                currentPath = window.location.pathname;
                setTimeout(updateFloatingNav, 100);
            }
        });
        
        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    }

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initFloatingNav);
    } else {
        initFloatingNav();
    }

    // Re-initialize on navigation (for MkDocs)
    if (typeof window !== 'undefined') {
        window.addEventListener('load', () => {
            setTimeout(updateFloatingNav, 500);
        });
    }

})();
