require 'find'

module Jekyll
  class DirectoryTreeTag < Liquid::Tag
    def initialize(tag_name, markup, tokens)
      super
      @path = markup.strip || '_notes'
    end

    def render(context)
      site = context.registers[:site]
      source = site.source
      base_path = File.join(source, @path)
      
      return "Directory not found: #{@path}" unless Dir.exist?(base_path)
      
      tree = generate_tree(base_path, @path)
      "```\n#{tree}```"
    end

    private

    def generate_tree(path, display_path, prefix = "", is_last = true)
      entries = Dir.entries(path).reject { |e| e.start_with?('.') || e == 'site' }
                   .sort_by { |e| [File.directory?(File.join(path, e)) ? 0 : 1, e.downcase] }
      
      result = ""
      
      # Add the current directory name
      if prefix.empty?
        result += "#{display_path}/\n"
      end
      
      entries.each_with_index do |entry, index|
        entry_path = File.join(path, entry)
        is_directory = File.directory?(entry_path)
        is_last_entry = index == entries.length - 1
        
        # Create the tree structure
        current_prefix = prefix + (is_last ? "└── " : "├── ")
        next_prefix = prefix + (is_last ? "    " : "│   ")
        
        if is_directory
          result += "#{current_prefix}#{entry}/\n"
          
          # Add description for key directories
          description = get_directory_description(entry)
          if description && entries.length < 10  # Only add descriptions for smaller directories
            result += "#{next_prefix.gsub('├── ', '│   ').gsub('└── ', '    ')}# #{description}\n"
          end
          
          # Recursively process subdirectories (limit depth to avoid huge output)
          if prefix.count('│') + prefix.count(' ') < 12
            subtree = generate_tree(entry_path, entry, next_prefix, is_last_entry)
            result += subtree if subtree && !subtree.empty?
          end
        else
          result += "#{current_prefix}#{entry}"
          
          # Add description for key files
          description = get_file_description(entry)
          result += "    # #{description}" if description
          result += "\n"
        end
      end
      
      result
    end

    def get_directory_description(dir_name)
      descriptions = {
        'assets' => 'Website styling and resources',
        'calculus_and_linear_algebra' => 'Mathematical foundations',
        'engineering_and_data_structure' => 'Programming & data structures',
        'ml_fundamentals' => 'Machine learning core concepts',
        'language_model' => 'Natural language processing',
        'neural_networks_and_deep_learning' => 'Deep learning',
        'probability_and_markov' => 'Probability & statistics',
        'javascripts' => 'Website functionality',
        'resources' => 'Reference materials',
        'images' => 'Image resources',
        'styles' => 'CSS styling files'
      }
      descriptions[dir_name]
    end

    def get_file_description(file_name)
      descriptions = {
        'index.md' => 'Main landing page',
        'Foundational knowledge plan.md' => 'Learning roadmap',
        'Information_Theory.md' => 'Information theory concepts',
        'Integration_and_Project.md' => 'Integration projects',
        'hero.css' => 'Hero section styling',
        'layout.css' => 'Main layout styling',
        'mathjax.js' => 'Mathematical equation rendering',
        'floating-nav.js' => 'Navigation enhancements',
        'Happy-LLM-v1.0.pdf' => 'Reference materials'
      }
      descriptions[file_name]
    end
  end
end

Liquid::Template.register_tag('directory_tree', Jekyll::DirectoryTreeTag)
