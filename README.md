# Gold AI Trading Scanner

## اجرا در سیستم شخصی
1. Python 3.10 یا جدیدتر نصب باشد.
2. در پوشه پروژه اجرا کنید:
   `pip install -r requirements.txt`
3. سپس:
   `streamlit run app.py`

## استقرار روی Render
1. فایل ZIP را باز و محتویات آن را در یک GitHub repository بارگذاری کنید.
2. در Render گزینه New + → Blueprint یا New Web Service را انتخاب کنید.
3. Repository را متصل کنید.
4. Build Command: `pip install -r requirements.txt`
5. Start Command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
6. سرویس را Deploy کنید.

## داده بازار
- نماد پیش‌فرض `XAUUSD=X` در Yahoo Finance است.
- `GC=F` قرارداد آتی طلای COMEX است و می‌تواند جایگزین باشد؛ این دو دقیقاً یک بازار/قیمت نیستند.
- دسترسی Yahoo Finance ممکن است در بعضی شبکه‌ها یا زمان‌ها محدود یا ناپایدار باشد.

## هشدار
این پروژه ابزار کمکی و آزمایشی است، نه توصیه مالی. امتیاز سیگنال درصد احتمال برد نیست. قبل از معامله واقعی بک‌تست و حساب دمو انجام دهید. حد سود/ضرر، حجم و اهرم باید با مشخصات قرارداد و قیمت بروکر خودتان تطبیق داده شود.
