document.addEventListener("DOMContentLoaded", function() {
    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(function(position) {
            var lat = position.coords.latitude;
            var lon = position.coords.longitude;

            // Send the location data to the server
            fetch('/get_location/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': '{{ csrf_token }}'
                },
                body: JSON.stringify({latitude: lat, longitude: lon})
            })
            .then(response => response.json())
            .then(data => {
                console.log('Location sent successfully:', data);
            })
            .catch((error) => {
                console.error('Error:', error);
            });
        });
    } else {
        console.log("Geolocation is not supported by this browser.");
    }
});