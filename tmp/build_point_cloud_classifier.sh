CATKIN_WS=/home/kirill/catkin_ws/
source $CATKIN_WS/devel/setup.bash --extend
DIR="$( cd "$(dirname "$0")"&& pwd)"
PROJECT_DIR=${DIR}/..
mkdir $PROJECT_DIR/build
cd $PROJECT_DIR/build
cmake -DCMAKE_INSTALL_PREFIX=$PROJECT_DIR/install ..
make
#make install -DCMAKE_BUILD_TYPE=Release -DSETUPTOOLS_DEB_LAYOUT=OFF
make install
#source $PROJECT_DIR/install/setup.bash --extend