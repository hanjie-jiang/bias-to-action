# Bias to Action — __Build intuition, then build models.__ 

## Notes for ML & AI Fundamentals that Minimizes Confusion

This is the Obsidian notes for ML & AI learnings purposes based on different sources that one could 
find with plain language and easy to understand content. The engineering section is being enhanced with 
systematic algorithmic foundations from MIT 6.006 Introduction to Algorithms, seamlessly integrated 
with existing practical implementations and ML applications. Resources are all included in the notes as a standalone package.

All the materials can also be found readable at GitPage URL (https://hanjie-jiang.github.io/bias-to-action/)

## Repository Structure

```
_notes/
├── index.md                              # Main landing page
├── Foundational knowledge plan.md        # Learning roadmap
├── Information_Theory.md                 # Information theory concepts
├── Integration_and_Project.md            # Integration projects
│
├── assets/                               # Website resources
│   ├── images/                              # General image resources
│   ├── ml_fundamentals/                     # ML fundamentals resources (moved from scattered locations)
│   │   ├── Screenshot 2025-08-05 at 8.22.11 PM.png
│   │   ├── Screenshot 2025-08-05 at 9.30.27 PM.png
│   │   ├── baimian-ml.pdf
│   │   ├── gini_index_1.png
│   │   ├── gini_index_2.png
│   │   └── p-r_curve.png
│   └── styles/
│       ├── hero.css                         # Hero section styling
│       └── layout.css                       # Main layout styling
│
├── calculus_and_linear_algebra/          # Mathematical Foundations
│   ├── Calculus_and_Linear_Algebra_Overview.md  # Mathematical foundations overview
│   ├── Linear_Algebra_for_ML.md             # Vectors, matrices, and ML operations
│   ├── Calculus_and_Gradient_Descent.md     # Optimization and gradient methods
│   └── Asymptotic_Analysis_Theory.md        # Complexity theory and algorithm analysis
│
├── engineering_and_data_structure/       # Programming & Data Structures
│   ├── Overview/
│   │   └── Engineering_and_Data_Structure_Overview.md
│   ├── Data_Structures/
│   │   ├── Arrays/
│   │   │   ├── Arrays_Overview.md               # Comprehensive arrays vs linked lists comparison
│   │   │   ├── Dynamic_Arrays.md                # Resizable arrays and Python lists
│   │   │   └── Array_Problems.md                # Array-based coding problems
│   │   ├── Linked_Lists/
│   │   │   ├── Linked_Lists_Overview.md         # Singly, doubly, circular linked lists theory
│   │   │   ├── Linked_List_Implementation.md    # Python implementation with node structures
│   │   │   └── Linked_List_Problems.md          # Two pointers, reversal, merging patterns
│   │   ├── Stacks/
│   │   │   ├── Stacks_Overview.md               # LIFO operations, applications, patterns
│   │   │   ├── Stack_Implementation.md          # Array vs linked list implementations
│   │   │   └── Stack_Problems.md                # Valid parentheses, monotonic stack, DFS
│   │   ├── Queues/
│   │   │   ├── Queues_Overview.md               # FIFO operations, variants (priority, deque)
│   │   │   ├── Queue_Implementation.md          # Array vs linked list implementations
│   │   │   └── Queue_Problems.md                # BFS, sliding window, level-order traversal
│   │   ├── Hash_Tables/
│   │   │   ├── Hash_Tables_Overview.md
│   │   │   ├── Hash_Functions_and_Collisions.md
│   │   │   ├── Python_Dictionaries.md
│   │   │   ├── Python_Dictionary_Operations.md
│   │   │   ├── Python_Sets.md
│   │   │   ├── Python_Set_Operations.md
│   │   │   └── Hash_Table_Problems.md
│   │   └── Recursion/
│   │       └── recursion_overview.md
│   ├── Algorithms/
│   │   ├── Search_Algorithms/
│   │   │   ├── Search_Algorithms_Overview.md
│   │   │   ├── Binary_Search_Fundamentals.md
│   │   │   ├── Binary_Search_Variations.md
│   │   │   └── Search_Problems.md              # Peak finding and search applications
│   │   └── Sorting_Algorithms/
│   │       ├── Sorting_Algorithms_Overview.md   # Comprehensive sorting theory with quicksort details
│   │       └── Sorting_Problems.md              # LeetCode problems and inversion counting
│   ├── Problem_Solving/
│   │   ├── Set_Dictionary_Problems/
│   │   │   ├── Array_Intersection.md
│   │   │   ├── Non_Repeating_Elements.md
│   │   │   ├── Unique_Elements.md
│   │   │   └── Anagram_Pairs.md
│   │   └── String_Problems/
│   │       ├── String_Operations.md
│   │       └── Unique_Strings.md
│   └── Resources/
│       ├── MIT_6006_Integration_Templates.md   # Templates for systematic algorithm integration
│       ├── Common_Patterns.md
│       ├── Time_Complexity_Guide.md
│       └── Interview_Strategies.md
│
├── ml_fundamentals/                      # Machine Learning Fundamentals
│   ├── ML_Fundamentals_Overview.md          # ML overview
│   ├── feature_engineering/
│   │   ├── categorical_encoding.md
│   │   ├── data_types_and_normalization.md
│   │   └── feature_crosses.md
│   ├── model_evaluation/
│   │   ├── evaluation_methods.md
│   │   ├── metrics_and_validation.md
│   │   ├── hyperparameter_tuning.md
│   │   └── resources/                       # Images and PDFs
│   ├── regularization/
│   │   ├── overfitting_underfitting.md
│   │   ├── l1_l2_regularization.md
│   │   └── early_stopping.md
│   ├── classical_algorithms/
│   │   ├── linear_regression.md
│   │   ├── logistic_regression.md
│   │   └── decision_trees.md
│   └── unsupervised_learning/
│       ├── k_nearest_neighbors.md
│       └── k_means_clustering.md
│
├── language_model/                       # Natural Language Processing
│   ├── Ngram_Language_Modeling.md           # N-gram models
│   └── resources/
│       └── Happy-LLM-v1.0.pdf              # Reference materials
│
├── neural_networks_and_deep_learning/    # Deep Learning
│   ├── Neural_Networks_and_Deep_Learning_Overview.md
│   ├── neural_networks_sections/
│   │   └── Introduction_to_Perceptron_Algorithm.md
│   └── resources/                           # Reference materials
│
├── probability_and_markov/               # Probability & Statistics
│   ├── Probability_and_Markov_Overview.md
│   ├── probability_and_markov_sections/
│   │   ├── conditional_probability_and_bayes_rule.md
│   │   ├── joint_and_marginal_distributions.md
│   │   └── naive_bayes_and_gaussian_naive_bayes.md
│   └── resources/
│       └── conditional_probability.png      # Diagrams and images
│
└── javascripts/                          # Website functionality
    ├── mathjax.js                           # Mathematical equation rendering
    └── floating-nav.js                      # Navigation enhancements
```

## Key Features

- **Comprehensive Coverage**: From basic probability to advanced neural networks with mathematical foundations
- **MIT 6.006 Integration**: Systematic algorithmic foundations seamlessly integrated with practical applications and theoretical depth
- **Mathematical Rigor**: Complete asymptotic analysis, complexity theory, and calculus connections for ML optimization
- **Complete Data Structures**: Arrays, linked lists, stacks, queues, and hash tables with implementation details and complexity analysis
- **Algorithms Foundations**: Search and sorting algorithms with step-by-step visualizations, quicksort/quickselect theory, and inversion counting
- **Linear Data Structures**: LIFO (stacks) and FIFO (queues) operations with real-world applications and problem patterns
- **Interconnected**: Cross-references and links between related topics across mathematics, algorithms, and ML applications  
- **Theory + Practice**: Rigorous mathematical analysis combined with hands-on implementations and real-world examples
- **Template-Driven**: Structured templates for consistent content organization and systematic expansion of algorithmic concepts
- **Visual Learning**: Mathematical equations (LaTeX), algorithm diagrams, complexity comparisons, and data structure visualizations
- **Modern UI**: Pastel-themed responsive design with hover dropdowns and optimized mathematical rendering
- **Searchable & Organized**: Full-text search across all content with centralized asset management and logical navigation structure

## Update logs

### version 2025-10-21

- **Queue Implementation Mastery**: Enhanced Queues Overview with comprehensive Python deque implementation and performance analysis
- **FIFO vs LIFO Comparison**: Added detailed comparison tables between queues, stacks, and priority queues with complexity analysis
- **BFS Algorithm Patterns**: Implemented breadth-first search patterns with queue-based traversal algorithms
- **Queue Variants Deep Dive**: Comprehensive coverage of simple queues, circular queues, priority queues, and deques with use cases
- **Data Structure Selection Guide**: Added decision matrix for when to use queues vs other data structures with practical examples
- **Performance Optimization**: Demonstrated why collections.deque outperforms list operations for queue implementations

### version 2025-10-20

- **Monotonic Stack Problems**: Added Daily Temperatures (LeetCode #739) with comprehensive solution using decreasing monotonic stack
- **Temperature Analysis Patterns**: Implemented both warmer and cooler temperature finding algorithms with O(n) time complexity
- **Stack Problem Collection**: Enhanced Stack Problems section with detailed explanations of monotonic stack techniques
- **_Algorithm Optimization**: Demonstrated how monotonic stacks improve from O(n²) brute force to O(n) optimal solutions
- **Problem Pattern Recognition**: Added examples showing "next greater/smaller element" problem variations

### version 2025-10-13

- **Complete Data Structures Suite**: Added comprehensive coverage of linked lists, stacks, and queues with dedicated folders and structured content
- **Linear Data Structures Mastery**: Detailed theory, implementations, and problem patterns for LIFO (stacks) and FIFO (queues) operations
- **Linked Lists Deep Dive**: Singly, doubly, and circular linked lists with implementation details and comparison to arrays
- **Stack Applications**: Valid parentheses, monotonic stacks, DFS algorithms, and expression evaluation with real problem solutions
- **Queue Variants**: Simple queues, circular queues, priority queues, and deques with BFS and sliding window applications
- **Navigation Integration**: Updated MkDocs navigation, front page cards, and cross-references for seamless learning flow
- **Implementation Focus**: Array-based vs linked list-based implementations with performance trade-offs and use case analysis

### version 2025-10-09

- **Sorting Algorithms Mastery**: Completed comprehensive sorting algorithms section with detailed quicksort step-by-step visualizations, quickselect algorithm analysis, and partitioning mechanics
- **Search & Sort Integration**: Finalized both search algorithms (binary search, peak finding) and sorting algorithms (quicksort, mergesort, inversion counting) with theoretical depth and practical implementations
- **Advanced Problem Solving**: Added LeetCode problems including Kth Largest Element with multiple approaches, inversion counting with merge sort, and sorting-based techniques
- **Algorithm Visualization**: Created detailed partitioning examples showing element-by-element moves and explaining why "partitioning ≠ sorting" concept
- **Cross-Reference Optimization**: Established proper links between sorting theory (Overview) and practical applications (Problems) for better learning flow

### version 2025-10-01
- **Fundamental Concepts Enhancement**: Added comprehensive "Interface vs Data Structure" distinction to Engineering & Data Structure Overview explaining the difference between abstract operations and concrete implementations
- **Python Sets Page Enhancement**: Updated Python Sets page title to "Sets and Python Sets Overview" for better clarity and scope representation
- **Reference Updates**: Updated all cross-references across Engineering & Data Structure Overview, Hash Tables Overview, and mkdocs.yml to maintain consistency with new page titles
- **Data Structure Comparison Table**: Implemented comprehensive comparison table showing time complexities for different data structures (Array, Sorted Array, Hash Table/Set, Binary Search Tree) across various operations (Build, Search, Insert/Delete, find_min(), find_prev())
- **Documentation Consistency**: Ensured all internal links and navigation references reflect updated page titles and content structure

### version 2025-09-29

- **Mathematical Foundations Continued**: Added comprehensive Mathematical Foundations section with Asymptotic Analysis Theory and enhanced Calculus & Gradient Descent
- **Arrays & Linked Lists Integration**: Created complete comparison guide with static arrays vs linked lists, time complexities, and implementation examples  
- **Enhanced Arrays Overview**: Merged comprehensive data structures content into unified Arrays Overview with visual comparisons and MIT 6.006 connections
- **Navigation Fixes**: Resolved MkDocs navigation issues and proper mathematical equation rendering across all content
- **Asset Reorganization**: Centralized all images and resources to `/assets/` directory structure for better organization
- **Content Optimization**: Fixed mathematical notation formatting, improved cross-references, and enhanced code examples with detailed explanations

### version 2025-09-28

- **MIT 6.006 Integration Framework**: Added structured templates for seamlessly integrating MIT Introduction to Algorithms content
- **Template System**: Created comprehensive content templates for consistent expansion of theoretical foundations
- **Enhanced Learning Path**: Designed integration strategy combining intuitive understanding with systematic rigor
- **Preserved Content**: All existing materials maintained while adding framework for algorithmic depth

### version 2025-08-25

- updated the `engineering_and_data_structure` folder with newly added recursion content
- added in the search and sort algorithm sections in appropriate folders
- updated the README.md format to describe the website structure

### version 2025-08-23

- restructured the `_notes\engineering_and_data_structure` section to make it self-contained
- reorganized Data Structures section under Hash Tables with comprehensive theory and Python implementations
- added Hash Tables Overview, Hash Functions and Collisions, and Hash Table Problems sections
- consolidated Python Sets and Python Dictionaries under the Hash Tables umbrella for better conceptual organization
- added in the `_notes\calculus_and_linear_algebra` section for fundamental mathematics review in the future
- updated the README.md format to describe the website structure

### version 2025-08-22

- added in the engineering and data structure related pages for future reference when coding
- refactored the front page design and made sure that the pastel hero looks ok
- fixed the math equations not showing properly but in raw format
- restructured the ML fundamentals section to be more organized and readable
- upgraded the front page to have hover over drop-down menu
