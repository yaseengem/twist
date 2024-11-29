
import React, { useState } from 'react';
import { View, Text, Button, TextInput } from 'react-native';
import { useDispatch } from 'react-redux';
import axios from 'axios';

export default function HomeScreen({ navigation }) {
  const [resume, setResume] = useState('');
  const [jobDescription, setJobDescription] = useState('');
  const dispatch = useDispatch();

  const handleSubmit = async () => {