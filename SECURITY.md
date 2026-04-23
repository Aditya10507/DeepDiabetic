# Security and Configuration Guide

## Environment Variables Required in Production

### SECRET_KEY (CRITICAL)
- **Required**: Yes, in production (DEBUG=False)
- **Default**: None (will raise error if missing in production)
- **Generation**: `python -c 'from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())'`
- **Guidelines**:
  - Keep SECRET_KEY secret (don't commit to version control)
  - Use `.env` file and add it to `.gitignore`
  - Rotate quarterly in production
  - Never use the development fallback key in production

### DEBUG
- **Production**: Must be `False`
- **Development**: Can be `True`
- **Default**: `False`
- **Danger**: Setting DEBUG=True in production exposes sensitive information

### ALLOWED_HOSTS
- **Production**: Must list all domain names/IPs serving the application
- **Example**: `ALLOWED_HOSTS=myapp.com,www.myapp.com,api.myapp.com`
- **Default**: Only localhost, 127.0.0.1, testserver
- **Render.io**: Set `RENDER_EXTERNAL_HOSTNAME` environment variable

## Database Security

### SQLite (Default, Development Only)
- Simple, file-based database
- Suitable for development and demo deployments
- Not recommended for production with concurrent users

### PostgreSQL (Recommended for Production)
```
DB_ENGINE=postgresql
DB_NAME=diabetic_db
DB_USER=db_user
DB_PASSWORD=secure_password_here
DB_HOST=database.example.com
DB_PORT=5432
DB_SSLMODE=require
```
- Set `DB_SSLMODE=require` for encrypted connections
- Use strong, randomly generated passwords
- Restrict database access to application server only

Azure Database for PostgreSQL is supported through the same settings or through Azure-provided values such as:

```
AZURE_POSTGRESQL_CONNECTIONSTRING=host=<server>.postgres.database.azure.com port=5432 dbname=<database> sslmode=require user=<username> password=<password>
```

### MySQL
```
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=mysql_user
MYSQL_PASSWORD=secure_password
MYSQL_DATABASE=diabetic_db
```

## Form Validation Improvements

### Contact Number Validation
- **Supports International Formats**:
  - Standard format: 10 digits
  - With country codes: +1-234-567-8901, 91-9876543210
  - With formatting: (234) 567-8901, +1 (234) 567-8901
  - Range: 9-15 digits after cleanup
  
- **Examples of Valid Formats**:
  - `9876543210` (India)
  - `+1-234-567-8901` (US with country code)
  - `(234) 567-8901` (US format)
  - `91 98765 43210` (India with spacing)
  - `+44-20-7946-0958` (UK)

## Authentication Security

### Email-based Authentication
- Users can login with email address instead of username
- Email must be unique (validated at registration)
- Passwords are hashed using Django's PBKDF2 algorithm

### Password Validation
Django's built-in validators enforce:
1. Password similarity check (not too similar to username/email)
2. Minimum length (8 characters)
3. Common password check (top 20,000 common passwords)
4. Numeric password check (not all digits)

### HTTPS/TLS (Production Only)
```python
# Automatically enabled when DEBUG=False
CSRF_COOKIE_SECURE = True      # Only send CSRF cookie over HTTPS
SESSION_COOKIE_SECURE = True   # Only send session cookie over HTTPS
SECURE_BROWSER_XSS_FILTER = True
SECURE_CONTENT_TYPE_NOSNIFF = True
```

## Error Handling and Logging

### ML Pipeline Error Handling
- Image processing fails gracefully with informative error messages
- Model loading falls back to legacy models if primary weights unavailable
- File I/O errors caught and reported to user

### Logging Configuration
```
/logs/debug.log - All debug level and above messages
Console output - Info level and above messages
```

## Deployment Checklist

- [ ] Set `SECRET_KEY` environment variable
- [ ] Set `DEBUG=False`
- [ ] Configure `ALLOWED_HOSTS` for your domain(s)
- [ ] Use PostgreSQL or MySQL (not SQLite) for production
- [ ] Enable HTTPS/TLS on your web server
- [ ] Set `DB_SSLMODE=require` for secure database connections
- [ ] Rotate SECRET_KEY quarterly
- [ ] Review logs regularly
- [ ] Keep Django and dependencies updated
- [ ] Use strong, unique database passwords
- [ ] Restrict database access to application server

## Contact Number Validation Migration

If migrating from the previous 10-digit-only validation:
- **Old behavior**: Only accepted exactly 10 digits
- **New behavior**: Accepts 9-15 digits with international formats
- **No data migration needed**: Previous valid phone numbers remain valid
- **Backwards compatible**: All previously valid formats still work

## Testing Security

Run the test suite to verify security:
```bash
python manage.py test DiabeticApp.tests
```

Tests include:
- Form validation (email uniqueness, international contact numbers)
- Authentication (email-based login)
- View security (login required for protected views)
- User profile management
