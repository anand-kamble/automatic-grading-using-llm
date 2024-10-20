
mkdir ~/.config &&\
mkdir ~/.config/ngrok &&\
touch ~/.config/ngrok/ngrok.yml &&\
echo "version: 2" > ~/.config/ngrok/ngrok.yml &&\
echo "authtoken: <your-authtoken-here>" >> ~/.config/ngrok/ngrok.yml &&\
./llava-v1.5-7b-q4.llamafile --server --nobrowser &\
ngrok http --domain=vastly-pleasing-sunbeam.ngrok-free.app 8080
