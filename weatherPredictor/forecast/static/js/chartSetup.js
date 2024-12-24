document.addEventListener('DOMContentLoaded', () => {
    const chartElement = document.getElementById('chart');
    if(!chartElement) {
        console.error('Canvas element not found');
        return;
    };

    const ctx = chartElement.getContext('2d');
    const gradient = ctx.createLinearGradient(0, -10, 0, 100);
    gradient.addColorStop(0, 'rgba(255, 0, 0, 1)');
    gradient.addColorStop(0, 'rgba(135, 255, 0, 1)');

    const forecastItems = document.querySelectorAll('.forecast-item');

    const temps = [];
    const times = [];

    forecastItems.forEach((item) => {
        const time = item.querySelector('.forecast-time').textContent;
        const temp = item.querySelector('.forecast-temperatureValue').textContent;
        const hum = item.querySelector('.forecast-humidityValue').textContent;

        if (time && temp && hum) {
            times.push(time);
            temps.push(temp);
        }
    });

    //Ensure all values are valid before using them
    if (times.length === 0 || temps.length === 0) {
        console.error('No valid data for time or temperature found.');
        return;
    }

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: times,
            datasets: [{
                label: 'Celsius Degrees',
                data: temps,
                borderColor: gradient,
                tension: 0.4,
                borderWidth: 2,
                pointRadius: 2,
            }],
        },
        options: {
            plugins: {
                legend: {
                    display: false,
                },
            },
            scales: {
                x: {
                    display: false,
                    grid: {
                        drawOnChartArea: false,
                    },
                },
                y: {
                    display: false,
                    grid: {
                        drawOnChartArea: false,
                    },  
                },
            },
            animation: {
                duration: 750,
            }
        }
    });
});