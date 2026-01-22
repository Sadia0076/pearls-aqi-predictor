def check_aqi_alert(aqi_value):
    if aqi_value >= 150:
        print("🚨 ALERT: Unhealthy AQI expected")
    elif aqi_value >= 200:
        print("🚨🚨 ALERT: Very Unhealthy AQI expected")
#For Later:Email ,SMS ,Slack, Push notification to telling aqi condition