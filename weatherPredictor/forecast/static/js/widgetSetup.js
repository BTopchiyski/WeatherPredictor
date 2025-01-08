document.addEventListener('DOMContentLoaded', function() {
    const city = "{{ city }}"; // Get the city from the template context

    window.myWidgetParam ? window.myWidgetParam : window.myWidgetParam = [];
    window.myWidgetParam.push({
        id: 21,
        cityid: '',
        appid: '7cf41a0173d30644b4db7cfd490c2cb9',
        units: 'metric',
        containerid: 'openweathermap-widget-21',
        city: city
    });

    (function() {
        var script = document.createElement('script');
        script.async = true;
        script.charset = "utf-8";
        script.src = "//openweathermap.org/themes/openweathermap/assets/vendor/owm/js/weather-widget-generator.js";
        var s = document.getElementsByTagName('script')[0];
        s.parentNode.insertBefore(script, s);
    })();
});