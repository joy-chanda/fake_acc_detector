const express = require('express');
const cors = require('cors');
const axios = require('axios');
require('dotenv').config();

const app = express();
app.use(cors());
app.use(express.static(__dirname));

// Internal endpoint that works without login if you spoof headers right
const INSTA_URL = 'https://i.instagram.com/api/v1/users/web_profile_info/?username=';

app.get('/api/scrape/:username', async (req, res) => {
  try {
    const { username } = req.params;

    // We must pass specific headers or Instagram blocks the request
    const response = await axios.get(`${INSTA_URL}${username}`, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'X-IG-App-ID': '936619743392459', // Required public app ID
        'Accept': '*/*',
        'Accept-Language': 'en-US,en;q=0.9',
        ...(process.env.IG_SESSIONID ? { 'Cookie': `sessionid=${process.env.IG_SESSIONID}` } : {})
      },
      timeout: 8000
    });

    const user = response.data?.data?.user;
    if (!user) {
      return res.status(404).json({ error: 'User not found or profile blocked' });
    }

    // Extract the exact 12 features our ML model needs
    const stats = {
      // 1. Follower ratio (followers / (followers + following))
      // Adding 1 to avoid division by zero
      edge_followed_by: user.edge_followed_by.count / (user.edge_followed_by.count + user.edge_follow.count + 1),
      
      // 2. Following ratio
      edge_follow: user.edge_follow.count / (user.edge_followed_by.count + user.edge_follow.count + 1),
      
      // 3 & 4. Username length and numbers
      username_length: user.username.length,
      username_has_number: /\d/.test(user.username) ? 1 : 0,
      
      // 5 & 6. Full name length and numbers
      full_name_length: user.full_name ? user.full_name.length : 0,
      full_name_has_number: user.full_name && /\d/.test(user.full_name) ? 1 : 0,
      
      // Binary flags
      is_private: user.is_private ? 1 : 0,
      is_joined_recently: user.is_joined_recently ? 1 : 0,
      has_channel: user.has_channel ? 1 : 0,
      is_business_account: user.is_business_account ? 1 : 0,
      has_guides: user.has_guides ? 1 : 0,
      has_external_url: user.external_url ? 1 : 0,
      
      // Extra UI data for the dashboard
      _display: {
        avatar: user.profile_pic_url_hd || user.profile_pic_url,
        name: user.full_name,
        bio: user.biography,
        followers: user.edge_followed_by.count,
        following: user.edge_follow.count,
        posts: user.edge_owner_to_timeline_media?.count || 0
      }
    };

    res.json(stats);
  } catch (error) {
    if (error.response && error.response.status === 404) {
      res.status(404).json({ error: 'Instagram account not found.' });
    } else {
      console.error('Scrape error:', error.message);
      res.status(500).json({ error: 'Instagram rate limited us or profile is completely hidden.' });
    }
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Fake IG Detector API running on http://localhost:${PORT}`);
});
