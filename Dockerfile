FROM nvidia/cuda:10.2-devel
ARG	folder_name=rational_rl
RUN 	apt-get update -y && apt-get install -y python3-pyqt5 python3-pip
RUN	mkdir /home/$folder_name

COPY	./requirements.txt /home/$folder_name

WORKDIR /home/$folder_name/
RUN	pip3 install --upgrade pip
RUN	pip3 install -r requirements.txt
RUN	pip3 install mushroom_rl[all]
