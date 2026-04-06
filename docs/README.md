# FlashRL Project Page

This directory contains the project page for FlashRL.

## File Structure

```
docs/
├── index.html         # Main project page
├── style.css          # Styling for the project page
├── README.md          # This file
├── images/            # Image assets (screenshots, diagrams, etc.)
├── videos/            # Video demonstrations
├── thumbnails/        # Video thumbnails
└── dataset/           # Dataset files or links
```

## Setup for GitHub Pages

To publish this project page on GitHub Pages:

1. Go to your repository settings
2. Navigate to "Pages" section
3. Under "Source", select the branch (e.g., `master` or `main`)
4. Select `/docs` as the folder
5. Save the settings

Your page will be available at: `https://<username>.github.io/<repository-name>/`

## Customization Guide

### Updating Content

1. **Title and Authors**: Edit the `<header>` section in `index.html`
2. **Links**: Update the navigation buttons with your paper, GitHub, and demo links
3. **Abstract**: Modify the `#abstract` section
4. **Features**: Add or remove features in the `#features` section
5. **Videos**: Add video files to the `videos/` folder and update the `#videos` section

### Adding Videos

1. Place your video files in the `videos/` folder
2. Optionally add thumbnails to the `thumbnails/` folder
3. Update the video grid in `index.html`:

```html
<div class="video-item">
    <video controls>
        <source src="videos/your-video.mp4" type="video/mp4">
    </video>
    <p class="video-caption">Your video description</p>
</div>
```

### Adding Images

Place images in the `images/` folder and reference them in your HTML:

```html
<img src="images/your-image.png" alt="Description">
```

## Local Testing

To test the page locally, you can use Python's built-in HTTP server:

```bash
cd docs
python -m http.server 8000
```

Then open `http://localhost:8000` in your browser.

## Styling

The page uses:
- **Font**: Inter (Google Fonts)
- **Color scheme**: Clean white background with dark text
- **Responsive design**: Works on desktop, tablet, and mobile
- **Animations**: Smooth transitions and hover effects

To customize colors and styles, edit `style.css`.
