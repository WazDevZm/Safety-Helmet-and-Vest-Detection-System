'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';

export default function Home() {
  const [currentSlide, setCurrentSlide] = useState(0);
  const [currentVerse, setCurrentVerse] = useState(0);

  const carouselSlides = [
    {
      image: '/images/hero-youth-group.jpg',
      title: 'Youth Worship Service',
      description: 'Join us for inspiring worship and fellowship'
    },
    {
      image: '/images/hero-pathfinder-camp.jpg',
      title: 'Pathfinder Camporee',
      description: 'Outdoor adventures and skill development'
    },
    {
      image: '/images/hero-community-service.jpg',
      title: 'Community Service',
      description: 'Making a difference in our community'
    },
    {
      image: '/images/hero-bible-study.jpg',
      title: 'Bible Study Groups',
      description: 'Growing together in faith and knowledge'
    }
  ];

  const bibleVerses = [
    {
      text: "Don't let anyone look down on you because you are young, but set an example for the believers in speech, in conduct, in love, in faith and in purity.",
      reference: "1 Timothy 4:12 (NIV)"
    },
    {
      text: "Trust in the Lord with all your heart and lean not on your own understanding; in all your ways submit to him, and he will make your paths straight.",
      reference: "Proverbs 3:5-6 (NIV)"
    },
    {
      text: "I can do all this through him who gives me strength.",
      reference: "Philippians 4:13 (NIV)"
    }
  ];

  useEffect(() => {
    const slideInterval = setInterval(() => {
      setCurrentSlide((prev) => (prev + 1) % carouselSlides.length);
    }, 5000);

    const verseInterval = setInterval(() => {
      setCurrentVerse((prev) => (prev + 1) % bibleVerses.length);
    }, 10000);

    return () => {
      clearInterval(slideInterval);
      clearInterval(verseInterval);
    };
  }, []);

  return (
    <div className="min-h-screen bg-aym-white">
      {/* Hero Section */}
      <section className="relative bg-gradient-to-br from-aym-green to-aym-yellow text-aym-white py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-12 items-center">
            {/* Hero Content */}
            <div className="space-y-8">
              <div className="space-y-4">
                <h1 className="text-4xl md:text-6xl font-bold text-aym-white">
                  Seventh-day Adventist Church
                </h1>
                <h2 className="text-2xl md:text-3xl font-semibold text-aym-yellow">
                  Adventist Youth Ministries
                </h2>
              </div>
              
              <div className="space-y-4">
                <h3 className="text-2xl font-semibold text-aym-white">
                  Welcome to Our Youth Community
                </h3>
                <p className="text-lg text-aym-white/90 leading-relaxed">
                  Join us in building faith, character, and service in young people worldwide. 
                  Discover your purpose, grow in Christ, and make a difference in your community 
                  through our vibrant youth programs.
                </p>
              </div>

              {/* Quick Action Buttons */}
              <div className="flex flex-wrap gap-4">
                <Link 
                  href="/about" 
                  className="bg-aym-yellow text-aym-green px-6 py-3 rounded-lg font-semibold hover:bg-aym-yellow-dark transition-colors duration-200"
                >
                  About Us
                </Link>
                <Link 
                  href="/contact#join-form" 
                  className="bg-aym-white text-aym-green px-6 py-3 rounded-lg font-semibold hover:bg-aym-yellow transition-colors duration-200"
                >
                  Join AY
                </Link>
                <Link 
                  href="/events" 
                  className="bg-aym-red text-aym-white px-6 py-3 rounded-lg font-semibold hover:bg-aym-red-dark transition-colors duration-200"
                >
                  Upcoming Events
                </Link>
                <Link 
                  href="/gallery" 
                  className="bg-aym-green text-aym-white px-6 py-3 rounded-lg font-semibold hover:bg-aym-green-dark transition-colors duration-200"
                >
                  Gallery
                </Link>
              </div>
            </div>

            {/* Carousel */}
            <div className="relative">
              <div className="relative h-96 rounded-lg overflow-hidden shadow-2xl">
                <div className="absolute inset-0 bg-gradient-to-t from-black/50 to-transparent z-10"></div>
                <img 
                  src={carouselSlides[currentSlide].image} 
                  alt={carouselSlides[currentSlide].title}
                  className="w-full h-full object-cover"
                />
                <div className="absolute bottom-0 left-0 right-0 p-6 text-aym-white z-20">
                  <h3 className="text-2xl font-bold mb-2">{carouselSlides[currentSlide].title}</h3>
                  <p className="text-lg">{carouselSlides[currentSlide].description}</p>
                </div>
              </div>
              
              {/* Carousel Indicators */}
              <div className="flex justify-center mt-4 space-x-2">
                {carouselSlides.map((_, index) => (
                  <button
                    key={index}
                    onClick={() => setCurrentSlide(index)}
                    className={`w-3 h-3 rounded-full transition-colors duration-200 ${
                      index === currentSlide ? 'bg-aym-yellow' : 'bg-aym-white/50'
                    }`}
                  />
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Weekly Bible Verse Section */}
      <section className="py-16 bg-aym-yellow-light">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <div className="bg-aym-white rounded-lg shadow-lg p-8 border-l-4 border-aym-green">
            <div className="flex items-center justify-center mb-4">
              <div className="w-12 h-12 bg-aym-green rounded-full flex items-center justify-center">
                <span className="text-aym-white text-xl">✝</span>
              </div>
            </div>
            <blockquote className="text-xl md:text-2xl font-medium text-gray-800 mb-4 italic">
              "{bibleVerses[currentVerse].text}"
            </blockquote>
            <cite className="text-aym-green font-semibold">
              {bibleVerses[currentVerse].reference}
            </cite>
            <div className="mt-6 space-x-4">
              <button 
                onClick={() => setCurrentVerse((prev) => (prev + 1) % bibleVerses.length)}
                className="bg-aym-green text-aym-white px-4 py-2 rounded-lg hover:bg-aym-green-dark transition-colors duration-200"
              >
                New Verse
              </button>
              <Link 
                href="/spiritual-corner" 
                className="text-aym-green hover:text-aym-green-dark font-semibold"
              >
                More Verses →
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Announcements Section */}
      <section className="py-16 bg-aym-white">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl md:text-4xl font-bold text-aym-green mb-4">
              This Week in AY
            </h2>
            <p className="text-lg text-gray-600">
              Stay updated with our latest announcements and activities
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {/* Featured Announcement */}
            <div className="lg:col-span-2 bg-gradient-to-br from-aym-green to-aym-yellow text-aym-white rounded-lg p-6 shadow-lg">
              <div className="flex items-center justify-between mb-4">
                <span className="bg-aym-red text-aym-white px-3 py-1 rounded-full text-sm font-semibold">
                  Featured
                </span>
                <span className="text-aym-yellow-light">March 15-17, 2024</span>
              </div>
              <h3 className="text-2xl font-bold mb-3">Youth Convention 2024 Registration Open</h3>
              <p className="text-lg mb-4">
                Join us for an inspiring weekend of worship, fellowship, and spiritual growth. 
                Early bird registration ends March 1st!
              </p>
              <Link 
                href="/events" 
                className="bg-aym-white text-aym-green px-6 py-3 rounded-lg font-semibold hover:bg-aym-yellow transition-colors duration-200 inline-block"
              >
                Register Now
              </Link>
            </div>

            {/* Regular Announcements */}
            <div className="space-y-6">
              <div className="bg-aym-white border-2 border-aym-yellow rounded-lg p-6 shadow-md hover:shadow-lg transition-shadow duration-200">
                <h3 className="text-xl font-semibold text-aym-green mb-2">Weekly AY Meeting</h3>
                <p className="text-aym-red font-medium mb-2">Every Saturday, 7:00 PM</p>
                <p className="text-gray-600 mb-3">
                  Join us for worship, Bible study, and fellowship. This week's speaker: Pastor Sarah Johnson
                </p>
                <Link href="/events" className="text-aym-green hover:text-aym-green-dark font-semibold">
                  Learn More →
                </Link>
              </div>

              <div className="bg-aym-white border-2 border-aym-green rounded-lg p-6 shadow-md hover:shadow-lg transition-shadow duration-200">
                <h3 className="text-xl font-semibold text-aym-green mb-2">Community Service Day</h3>
                <p className="text-aym-red font-medium mb-2">March 22, 2024</p>
                <p className="text-gray-600 mb-3">
                  Make a difference in our community through various service projects. All ages welcome!
                </p>
                <Link href="/events" className="text-aym-green hover:text-aym-green-dark font-semibold">
                  Join Us →
                </Link>
              </div>
            </div>
          </div>

          {/* Bulletin Download */}
          <div className="mt-12 text-center bg-aym-gray-light rounded-lg p-8">
            <h3 className="text-2xl font-bold text-aym-green mb-4">Weekly Bulletin</h3>
            <p className="text-lg text-gray-600 mb-6">
              Download this week's AY bulletin with all the latest news and announcements.
            </p>
            <button className="bg-aym-red text-aym-white px-8 py-3 rounded-lg font-semibold hover:bg-aym-red-dark transition-colors duration-200">
              📄 Download PDF
            </button>
          </div>
        </div>
      </section>
    </div>
  );
}