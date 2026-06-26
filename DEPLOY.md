# نشر بوت ICS على Railway أو Render (تشغيل 24/7)

بوت ICS يعمل بـ **Telegram long-polling** — أي أنه «عامل خلفي» (worker) لا يحتاج
منفذ ويب ولا دومين. هذا الدليل يشرح نشره بحيث يعمل دائمًا دون جهازك.

> ⚠️ **قاعدتان حرجتان قبل البدء**
> 1. **شغّل نسخة واحدة فقط من البوت على نفس التوكن.** نسختان = خطأ
>    `Conflict: terminated by other getUpdates`. **أوقف البوت على جهازك** قبل النشر:
>    `pkill -f "app.main bot"`.
> 2. **لا ترفع ملف `.env` إطلاقًا** (موجود في `.gitignore`). التوكن يُضبط من لوحة المنصة.

---

## 0) قرارات سريعة

| القرار | الخيار الأبسط | الخيار الأمتن |
|---|---|---|
| المنصة | **Railway** (worker دائم، إعداد أسهل) | Render (Background Worker مدفوع) |
| التخزين | **Volume دائم** + SQLite | Postgres مُدارة |

> 💸 **التكلفة بصراحة:** التشغيل الدائم مدفوع على الاثنتين عمليًا:
> Render Background Worker ≈ **$7/شهر**، Railway بالاستخدام ≈ **$5/شهر** بعد رصيد التجربة.
> الخطط المجانية تُنيّم الخدمة أو لا تدعم workers، فلا تصلح لبوت دائم.

---

## 1) جهّز المستودع على GitHub

من داخل مجلد `ics/` (اجعله جذر المستودع):

```bash
cd /Users/jabbar/Desktop/jab1000/ics
git init
git add .
git status            # تأكّد أن .env و *.db غير مدرجة
git commit -m "ICS bot — deployment ready"
```

أنشئ مستودعًا فارغًا على GitHub ثم:

```bash
git remote add origin https://github.com/<اسمك>/ics-bot.git
git branch -M main
git push -u origin main
```

تأكّد أن هذه الملفات موجودة في الجذر: `requirements.txt` · `Procfile` ·
`Dockerfile` · `runtime.txt` · `render.yaml` · `config.yaml` · `app/`.

---

## المسار A — Railway (موصى به)

### أ) عبر واجهة GitHub
1. ادخل <https://railway.app> → **New Project → Deploy from GitHub repo** → اختر `ics-bot`.
2. Railway يكتشف Python تلقائيًا ويشغّل أمر `Procfile`: `python -m app.main bot`.
   (إن لم يلتقطه، اضبط **Custom Start Command** = `python -m app.main bot`.)
3. **Variables** → أضِف:
   - `TELEGRAM_BOT_TOKEN` = توكن البوت
   - `TELEGRAM_ALLOWED_USER_IDS` = رقمك (مثال `123456789`)
   - `DATABASE_URL` = `sqlite:////data/ics.db`
4. **Volume**: اضغط على الخدمة → **+ Volume** → Mount path = `/data`.
   (هكذا تبقى قاعدة البيانات بين عمليات النشر.)
5. تأكّد أن **Replicas = 1** (الافتراضي).
6. **Deploy** ثم تابع **Logs** — يجب أن ترى:
   `ICS Telegram bot starting (paper-only, 1 authorized users).`

### ب) عبر Railway CLI (بدون GitHub)
```bash
npm i -g @railway/cli
railway login
cd /Users/jabbar/Desktop/jab1000/ics
railway init
railway up                         # يرفع المجلد الحالي ويبنيه
railway variables --set TELEGRAM_BOT_TOKEN=xxxx \
                  --set TELEGRAM_ALLOWED_USER_IDS=123456789 \
                  --set DATABASE_URL=sqlite:////data/ics.db
# أضف Volume على /data من اللوحة، ثم أعد النشر.
```

---

## المسار B — Render (Blueprint)

1. ادخل <https://render.com> → **New → Blueprint** → اربط مستودع `ics-bot`.
2. Render يقرأ `render.yaml` ويُنشئ خدمة **worker** اسمها `ics-bot` مع قرص دائم على `/var/data`.
3. في إعداد الخدمة، عيّن قيم الأسرار (لأنها `sync: false`):
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_ALLOWED_USER_IDS`
   - (`DATABASE_URL` مضبوط مسبقًا على `sqlite:////var/data/ics.db`)
4. **Create** → تابع **Logs** للتأكد من بدء التشغيل.

> إن لم تستخدم Blueprint: **New → Background Worker**، Build = `pip install -r requirements.txt`،
> Start = `python -m app.main bot`، وأضف **Disk** على `/var/data`، واضبط `DATABASE_URL` نفسه.

---

## 2) متغيرات البيئة المطلوبة (تُضبط من لوحة المنصة فقط)

| المتغير | القيمة |
|---|---|
| `TELEGRAM_BOT_TOKEN` | توكن البوت من @BotFather |
| `TELEGRAM_ALLOWED_USER_IDS` | أرقام المستخدمين المصرّح لهم، مفصولة بفواصل |
| `DATABASE_URL` | `sqlite:////data/ics.db` (Railway) أو `sqlite:////var/data/ics.db` (Render) أو رابط Postgres |

`pydantic-settings` يقرأ هذه المتغيّرات من بيئة المنصة مباشرةً (لا حاجة لملف `.env` هناك).

---

## 3) خيار Postgres الدائم (بديل أمتن للـ Volume)

1. أنشئ قاعدة Postgres مُدارة (Railway: **+ New → Database → PostgreSQL** / Render: **New → Postgres**).
2. أضِف المُحرّك إلى `requirements.txt`:
   ```
   psycopg[binary]>=3.1
   ```
3. اضبط `DATABASE_URL` على الرابط بصيغة SQLAlchemy:
   ```
   postgresql+psycopg://USER:PASSWORD@HOST:PORT/DBNAME
   ```
4. أعد النشر. الطبقة المستودعية (repository layer) في المشروع جاهزة لـ Postgres دون تغيير كود.

---

## 4) تحقّق أن البوت يعمل
- في **Logs**: سطر `Telegram command menu registered (16 commands).` وعدم وجود `Conflict`.
- في تيليجرام: أرسل `/status` ثم `/today`.
- **أوقف نسخة جهازك** إن كانت شغّالة: `pkill -f "app.main bot"` (وإلا ظهر `Conflict`).

---

## 5) استكشاف الأخطاء
| العَرَض | السبب / الحل |
|---|---|
| `Conflict: terminated by other getUpdates` | نسختان تستطلعان نفس التوكن. أبقِ واحدة فقط (Replicas=1 + أوقف المحلية). |
| كل الأوامر ترد `غير مصرّح لك` | `TELEGRAM_ALLOWED_USER_IDS` فارغ أو رقمك غير صحيح. |
| البيانات تُمسح بعد كل نشر | لم تربط Volume/Disk أو `DATABASE_URL` يشير لمسار غير دائم. استخدم `/data` المربوط أو Postgres. |
| فشل البناء على إصدار بايثون | تأكّد من `runtime.txt` (Render) و`.python-version` (Railway) = 3.11. |
| فشل تثبيت حزمة | ثبّت إصدارات `requirements.txt` على ما اختُبر محليًا، أو استخدم صورة `Dockerfile`. |

---

## ملخص الأوامر
```bash
# تجهيز ودفع
cd ics && git init && git add . && git commit -m "deploy" && git push -u origin main

# إيقاف البوت المحلي قبل/بعد النشر (مهم لتجنّب Conflict)
pkill -f "app.main bot"
```
