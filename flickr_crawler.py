import flickr_api

flickr_api.set_keys(api_key = '38aebeef4e98f69fdd90f52b01819c23', api_secret = '04443e0373405fbf')

a = flickr_api.auth.AuthHandler() #creates the AuthHandler object
perms = "read" # set the required permissions
url = a.get_authorization_url(perms)
print url

#username = 'chanran kim'
#user = flickr_api.Person.findByUserName(username)
#photos = user.getPublicPhotos() # otherwise
#print photos.info.pages # the number of available pages of results
#print photos.info.page  # the current page number
#print photos.info.total # total number of photos
# Get the title of the photos
#for photo in photos:
#    filename = photo['title'] + '.jpg'
#    print filename
#    photo.save(filename, size_label = 'Medium 640')

#flickr_api.Photo.search(tage = "rome")
photos = flickr_api.Photo.getRecent()

i=0
for photo in photos:
    i=i+1
    #print photo['title']
    #filename = photo['title'] + '.jpg'
    filename = str(i) + '.jpg'
    print filename
    photo.save(filename, size_label = 'Medium 640')
