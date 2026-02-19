# Use Ubuntu as the base image
FROM ubuntu:20.04

# Install necessary packages including MPICH and SSH
RUN apt-get update && apt-get install -y mpich openssh-server

# Set up passwordless SSH between containers
RUN mkdir /root/.ssh && \
    ssh-keygen -t rsa -N "" -f /root/.ssh/id_rsa && \
    cat /root/.ssh/id_rsa.pub >> /root/.ssh/authorized_keys && \
    echo "Host *\n    StrictHostKeyChecking no" >> /root/.ssh/config

# Ensure the SSH service can start
RUN mkdir /var/run/sshd

# Expose the SSH port
EXPOSE 22

# Copy application files
WORKDIR /app
COPY . /app

# Compile the MPI program
COPY mpi_program.c .
RUN mpicc -o mpi_program mpi_program.c

# Start SSH and keep the container running
CMD ["/usr/sbin/sshd", "-D"]
