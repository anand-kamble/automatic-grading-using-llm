# Use a base image, for example, Ubuntu
FROM ubuntu:latest

# Copy the file from the host machine to the container
COPY llava-v1.5-7b-q4.llamafile /llava-v1.5-7b-q4.llamafile


# Make the file executable (if needed)
RUN chmod +x /llava-v1.5-7b-q4.llamafile

RUN  apt-get update && apt install -y curl && curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null && echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | tee /etc/apt/sources.list.d/ngrok.list && apt update && apt install ngrok

COPY run_server.sh /run_server.sh
# Define the command to run the file
# CMD ["/file_in_container"]
CMD ["/bin/bash", "/run_server.sh"]
