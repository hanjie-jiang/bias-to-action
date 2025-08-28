/**
 * Main application logic for the notes index page
 * Handles folder cards, search functionality, and hero animations
 */

(function() {
  'use strict';

  // Wait for DOM and notes data to be ready
  function initializeApp() {
    const notes = window.ALL_NOTES || [];
    
    if (!notes.length) {
      console.warn('No notes data found');
      return;
    }

    // Build top-level folder groups from path first segment
    const groups = {};
    notes.forEach(note => {
      const cleanPath = (note.path || '').replace(/\/+$/, '');
      const pathParts = cleanPath.split('/');
      const topLevel = (pathParts.length > 1) ? pathParts[0] : '(root)';
      
      if (!groups[topLevel]) {
        groups[topLevel] = [];
      }
      groups[topLevel].push(note);
    });

    // Render folder cards
    renderFolderCards(groups);
    
    // Render all notes list
    const sortedNotes = notes.slice().sort((a, b) => a.path.localeCompare(b.path));
    renderNotesList(sortedNotes);
    
    // Initialize search functionality
    initializeSearch(sortedNotes);
    
    // Initialize hero animations
    initializeHeroAnimations();
  }

  function renderFolderCards(groups) {
    const cardsContainer = document.getElementById('folder-cards');
    if (!cardsContainer) return;

    const folderNames = Object.keys(groups).sort((a, b) => a.localeCompare(b));
    
    folderNames.forEach(folderName => {
      const folderNotes = groups[folderName]
        .slice()
        .sort((a, b) => a.path.localeCompare(b.path));
      
      const card = createFolderCard(folderName, folderNotes);
      cardsContainer.appendChild(card);
    });
  }

  function createFolderCard(folderName, notes) {
    const card = document.createElement('div');
    card.className = 'folder-card';
    
    // Header
    const header = document.createElement('h3');
    header.textContent = folderName;
    
    // Meta info
    const meta = document.createElement('div');
    meta.className = 'muted';
    meta.textContent = `${notes.length} ${notes.length === 1 ? 'note' : 'notes'}`;
    
    // Notes list (initially hidden)
    const notesList = document.createElement('ul');
    notesList.className = 'list';
    notesList.style.display = 'none';
    
    notes.forEach(note => {
      const listItem = document.createElement('li');
      
      const link = document.createElement('a');
      link.href = note.url;
      link.textContent = note.title || note.path;
      
      const pathInfo = document.createElement('small');
      pathInfo.textContent = note.path;
      
      listItem.appendChild(link);
      listItem.appendChild(pathInfo);
      notesList.appendChild(listItem);
    });
    
    // Toggle functionality
    card.addEventListener('click', () => {
      const isVisible = notesList.style.display !== 'none';
      notesList.style.display = isVisible ? 'none' : 'block';
    });
    
    card.appendChild(header);
    card.appendChild(meta);
    card.appendChild(notesList);
    
    return card;
  }

  function renderNotesList(notes) {
    const listContainer = document.getElementById('all-list');
    if (!listContainer) return;

    listContainer.innerHTML = '';
    
    notes.forEach(note => {
      const listItem = document.createElement('li');
      
      const link = document.createElement('a');
      link.href = note.url;
      link.textContent = note.title || note.path;
      
      const pathInfo = document.createElement('small');
      pathInfo.textContent = note.path;
      
      listItem.appendChild(link);
      listItem.appendChild(pathInfo);
      listContainer.appendChild(listItem);
    });
  }

  function initializeSearch(allNotes) {
    const searchInput = document.getElementById('search');
    const listContainer = document.getElementById('all-list');
    
    if (!searchInput || !listContainer) return;

    let currentSelection = -1;

    // Search functionality
    searchInput.addEventListener('input', () => {
      const query = searchInput.value.trim().toLowerCase();
      currentSelection = -1;
      
      if (!query) {
        renderNotesList(allNotes);
        return;
      }
      
      const filteredNotes = allNotes.filter(note => {
        const titleMatch = note.title && note.title.toLowerCase().includes(query);
        const pathMatch = note.path && note.path.toLowerCase().includes(query);
        return titleMatch || pathMatch;
      });
      
      renderNotesList(filteredNotes);
    });

    // Keyboard navigation
    searchInput.addEventListener('keydown', (event) => {
      const links = Array.from(listContainer.querySelectorAll('a'));
      if (!links.length) return;

      switch (event.key) {
        case 'ArrowDown':
          event.preventDefault();
          currentSelection = Math.min(links.length - 1, currentSelection + 1);
          links[currentSelection].focus();
          break;
          
        case 'ArrowUp':
          event.preventDefault();
          currentSelection = Math.max(0, currentSelection - 1);
          links[currentSelection].focus();
          break;
          
        case 'Enter':
          if (currentSelection >= 0) {
            links[currentSelection].click();
          }
          break;
      }
    });
  }

  function initializeHeroAnimations() {
    // Only add motion effects if user hasn't requested reduced motion
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      return;
    }

    const hero = document.querySelector('.hero');
    if (!hero) return;

    // Subtle parallax effect on mouse move
    hero.addEventListener('mousemove', (event) => {
      const rect = hero.getBoundingClientRect();
      const x = (event.clientX - rect.left) / rect.width - 0.5;
      const y = (event.clientY - rect.top) / rect.height - 0.5;
      
      // Subtle movement
      hero.style.transform = `translate(${x * 3}px, ${y * 3}px)`;
    });

    // Reset position when mouse leaves
    hero.addEventListener('mouseleave', () => {
      hero.style.transform = 'translate(0, 0)';
    });
  }

  // Initialize when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeApp);
  } else {
    initializeApp();
  }

})();
