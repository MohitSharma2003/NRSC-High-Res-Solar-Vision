// starfield.js
var canvas = document.getElementById('starfield');
var context = canvas.getContext('2d');

// Set canvas size to window size
function resizeCanvas() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
}

// Initial resize
resizeCanvas();

// Resize canvas when window is resized
window.addEventListener('resize', resizeCanvas);

// Star properties
var stars = [];
var numStars = 200;
var speed = 0.5;

// Create stars
for (var i = 0; i < numStars; i++) {
    stars.push({
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        z: Math.random() * 1000
    });
}

// Animation function
function animate() {
    context.fillStyle = 'rgba(0, 0, 0, 0.1)';
    context.fillRect(0, 0, canvas.width, canvas.height);
    
    context.fillStyle = 'white';
    context.beginPath();
    
    for (var i = 0; i < numStars; i++) {
        var star = stars[i];
        
        // Move star closer
        star.z -= speed;
        
        // Reset star if it's too close
        if (star.z <= 0) {
            star.z = 1000;
            star.x = Math.random() * canvas.width;
            star.y = Math.random() * canvas.height;
        }
        
        // Project star position
        var k = 128.0 / star.z;
        var px = star.x * k + canvas.width / 2;
        var py = star.y * k + canvas.height / 2;
        
        // Draw star if it's in view
        if (px >= 0 && px <= canvas.width && py >= 0 && py <= canvas.height) {
            var size = (1 - star.z / 1000) * 3;
            context.moveTo(px, py);
            context.arc(px, py, size, 0, Math.PI * 2);
        }
    }
    
    context.fill();
    requestAnimationFrame(animate);
}

// Start animation
animate();