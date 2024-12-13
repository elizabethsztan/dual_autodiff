# Use the official Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install Jupyter and upgrade pip
RUN pip install --upgrade pip \
    && pip install jupyter notebook

# Copy wheel files and install them
COPY wheels/ /app/wheels/
RUN pip install /app/wheels/dual_autodiff-0.1.0-py3-none-any.whl
RUN pip install /app/wheels/dual_autodiff_x-0.1.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

# Copy the notebooks
COPY notebooks/ /app/notebooks/

# Expose the default Jupyter port
EXPOSE 8888

# Run Jupyter Notebook server
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser", "--notebook-dir=/app/notebooks"]
