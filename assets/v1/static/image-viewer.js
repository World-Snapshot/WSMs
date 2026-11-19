/* Local magnifier and large-image viewer for project-page figures. */
(function () {
    'use strict';

    const container = document.querySelector('.container');
    if (!container) return;

    const imageSelector = 'img[alt]:not([alt=""])';
    const supportsHover = window.matchMedia('(hover: hover) and (pointer: fine)').matches;
    const lens = document.createElement('div');
    const lightbox = document.createElement('div');
    const panel = document.createElement('div');
    const largeImage = document.createElement('img');
    const closeButton = document.createElement('button');
    let lastFocusedElement = null;

    lens.className = 'image-magnifier-lens';
    lens.hidden = true;
    lens.setAttribute('aria-hidden', 'true');

    lightbox.className = 'image-lightbox';
    lightbox.hidden = true;
    lightbox.setAttribute('role', 'dialog');
    lightbox.setAttribute('aria-modal', 'true');
    lightbox.setAttribute('aria-label', 'Large image preview');

    panel.className = 'image-lightbox-panel';
    largeImage.className = 'image-lightbox-image';
    largeImage.alt = '';

    closeButton.type = 'button';
    closeButton.className = 'image-lightbox-close';
    closeButton.setAttribute('aria-label', 'Close large image preview');
    closeButton.title = 'Close';
    closeButton.innerHTML = '<i class="fas fa-times" aria-hidden="true"></i>';

    panel.appendChild(largeImage);
    panel.appendChild(closeButton);
    lightbox.appendChild(panel);
    document.body.appendChild(lens);
    document.body.appendChild(lightbox);

    function hideLens() {
        lens.hidden = true;
    }

    function updateLens(event, image) {
        if (!supportsHover || lightbox.hidden === false) return;

        const rect = image.getBoundingClientRect();
        const relativeX = (event.clientX - rect.left) / rect.width;
        const relativeY = (event.clientY - rect.top) / rect.height;
        if (relativeX < 0 || relativeX > 1 || relativeY < 0 || relativeY > 1) {
            hideLens();
            return;
        }

        const lensSize = 180;
        const zoom = 2.35;
        const backgroundWidth = rect.width * zoom;
        const backgroundHeight = rect.height * zoom;
        const source = image.currentSrc || image.src;

        lens.style.left = event.clientX + 'px';
        lens.style.top = event.clientY + 'px';
        lens.style.backgroundImage = 'url(' + JSON.stringify(source) + ')';
        lens.style.backgroundSize = backgroundWidth + 'px ' + backgroundHeight + 'px';
        lens.style.backgroundPosition =
            (lensSize / 2 - relativeX * backgroundWidth) + 'px ' +
            (lensSize / 2 - relativeY * backgroundHeight) + 'px';
        lens.hidden = false;
    }

    function closeLightbox() {
        if (lightbox.hidden) return;
        lightbox.hidden = true;
        largeImage.removeAttribute('src');
        largeImage.alt = '';
        document.documentElement.classList.remove('image-lightbox-open');
        if (lastFocusedElement) lastFocusedElement.focus();
    }

    function openLightbox(image, trigger) {
        hideLens();
        lastFocusedElement = trigger;
        largeImage.src = image.currentSrc || image.src;
        largeImage.alt = image.alt;
        lightbox.hidden = false;
        document.documentElement.classList.add('image-lightbox-open');
        closeButton.focus();
    }

    function positionOpenButton(image, host, button) {
        const imageRect = image.getBoundingClientRect();
        const hostRect = host.getBoundingClientRect();
        button.style.top = (imageRect.top - hostRect.top + 10) + 'px';
        button.style.left = (imageRect.right - hostRect.left - 10) + 'px';
    }

    function initializeImage(image) {
        if (image.dataset.imageViewerReady || image.closest('.image-lightbox')) return;

        let host = image.closest('figure');
        if (!host) {
            host = document.createElement('span');
            host.style.display = 'inline-block';
            image.parentNode.insertBefore(host, image);
            host.appendChild(image);
        }

        const openButton = document.createElement('button');
        openButton.type = 'button';
        openButton.className = 'image-viewer-open';
        openButton.setAttribute('aria-label', 'Open large preview of ' + image.alt);
        openButton.title = 'Open large preview';
        openButton.innerHTML = '<i class="fas fa-expand" aria-hidden="true"></i>';

        image.dataset.imageViewerReady = 'true';
        host.classList.add('image-viewer-host');
        host.appendChild(openButton);

        const updateButtonPosition = () => positionOpenButton(image, host, openButton);
        image.addEventListener('load', updateButtonPosition);
        image.addEventListener('mouseenter', updateButtonPosition);
        image.addEventListener('mousemove', event => updateLens(event, image));
        image.addEventListener('mouseleave', hideLens);
        openButton.addEventListener('click', event => {
            event.preventDefault();
            event.stopPropagation();
            openLightbox(image, openButton);
        });

        requestAnimationFrame(updateButtonPosition);
    }

    function initializeImages(root) {
        if (root.matches && root.matches(imageSelector)) initializeImage(root);
        if (root.querySelectorAll) root.querySelectorAll(imageSelector).forEach(initializeImage);
    }

    closeButton.addEventListener('click', closeLightbox);
    lightbox.addEventListener('click', event => {
        if (event.target === lightbox) closeLightbox();
    });
    document.addEventListener('keydown', event => {
        if (event.key === 'Escape') closeLightbox();
    });
    window.addEventListener('resize', () => {
        container.querySelectorAll(imageSelector).forEach(image => {
            const host = image.closest('.image-viewer-host');
            const button = host && host.querySelector(':scope > .image-viewer-open');
            if (host && button) positionOpenButton(image, host, button);
        });
        hideLens();
    });

    initializeImages(container);
    new MutationObserver(mutations => {
        mutations.forEach(mutation => {
            mutation.addedNodes.forEach(node => {
                if (node.nodeType === Node.ELEMENT_NODE) initializeImages(node);
            });
        });
    }).observe(container, { childList: true, subtree: true });
}());
