
echo "Installing the node modules"
npm install --prefer-offline --network-concurrency=20


echo "Running the app on web"
npx expo start --web