sudo mkdir /var/log/gemma3
sudo cp gemma3.service /etc/systemd/system
sudo systemctl enable gemma3
sudo systemctl start gemma3
