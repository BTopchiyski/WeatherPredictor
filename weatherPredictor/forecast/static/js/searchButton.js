document.addEventListener('DOMContentLoaded', function() {
    const searchButton = document.querySelector('.search-button');
    const geoInput = document.querySelector('.geo-input');

    if (searchButton && geoInput) {
        searchButton.addEventListener('click', function(event) {
            if (!geoInput.value) {
                event.preventDefault();
                alert('Please enter a city name.');
            }
        });
    }

    // Display error message if present
    const errorMessage = document.querySelector('.error-message');
    if (errorMessage) {
        alert(errorMessage.textContent);
    }
});