import flickr_api

def safe_name(filename):
    safe_string = str()
    for c in filename:
        if c.isalnum() or c in [' ','.','/']:
            safe_string = safe_string + c

        while safe_string.count("../"):
            # I use a loop because only replacing once would 
            # leave a hole in that a bad guy could enter ".../"
            # which would be replaced to "../" so the loop 
            # prevents tricks like this!
            safe_string = safe_string.replace("../","./")
        # Get rid of leading "./" combinations...
        safe_string = safe_string.lstrip("./")
    return safe_string


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
#photos = flickr_api.Photo.getRecent()

i=0

extras='url_c'
photos = flickr_api.Photo.search(tags = "korea", extras = extras)

for photo in photos:
    i=i+1
    #print photo['title']
    #filename = photo['title'] + '.jpg'
    filename = str(i) + '.jpg'
    print filename
    safe_string = safe_name(filename)
    valid = safe_string == filename
    if valid:
        try:
            photo.save(filename, size_label = "Medium 800")
        except:
            print "     error"



    
