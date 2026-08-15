document.addEventListener('DOMContentLoaded', function () {
  var images = document.querySelectorAll('.figure img');
  if (!images.length) return;

  var overlay = document.createElement('div');
  overlay.className = 'lightbox-overlay';
  overlay.innerHTML = '<span class="lightbox-close" aria-label="Close">&times;</span><img class="lightbox-img" alt="">';
  document.body.appendChild(overlay);

  var lightboxImg = overlay.querySelector('.lightbox-img');
  var closeBtn = overlay.querySelector('.lightbox-close');

  function openLightbox(src, alt) {
    lightboxImg.src = src;
    lightboxImg.alt = alt || '';
    overlay.classList.add('open');
    document.body.style.overflow = 'hidden';
  }

  function closeLightbox() {
    overlay.classList.remove('open');
    document.body.style.overflow = '';
    lightboxImg.src = '';
  }

  images.forEach(function (img) {
    img.classList.add('zoomable');
    img.addEventListener('click', function () {
      openLightbox(img.getAttribute('src'), img.getAttribute('alt'));
    });
  });

  overlay.addEventListener('click', function (e) {
    if (e.target === overlay || e.target === closeBtn) closeLightbox();
  });

  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape') closeLightbox();
  });
});
