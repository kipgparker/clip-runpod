# Base image
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

COPY builder /builder
RUN ls -la /builder

# Install Python dependencies (Worker Template)
COPY builder/requirements.txt /requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

# Cache Models
COPY builder/cache_models.py /cache_models.py
RUN python /cache_models.py && \
    rm /cache_models.py

# Add src files (Worker Template)
COPY src .
CMD python -u rp_handler.py