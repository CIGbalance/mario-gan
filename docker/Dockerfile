FROM pytorch/pytorch

RUN pip install future scipy pandas

# Install OpenJDK-8
RUN apt-get update && \
    apt-get install -y openjdk-8-jdk && \
    apt-get install -y ant && \
    apt-get clean;
    
# Fix certificate issues
RUN apt-get update && \
    apt-get install ca-certificates-java && \
    apt-get clean && \
    update-ca-certificates -f;

# Setup JAVA_HOME -- useful for docker commandline
ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/
ENV MKL_SERVICE_FORCE_INTEL=1
RUN export JAVA_HOME
RUN export MKL_SERVICE_FORCE_INTEL
