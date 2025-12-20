from flask import Blueprint, render_template, request, redirect, url_for, abort, flash, current_app, send_from_directory, send_file
from flask_login import login_required, current_user
from . import db
from .models import Videos
from werkzeug.utils import secure_filename
from .voicet import translate_video
import yt_dlp

import os
import random
import string


main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html', active='home')

@main.route('/profile')
@login_required
def profile():
    return render_template('profile.html', name=current_user.name, active='profile')

@main.route('/upload', methods=['POST'])
@login_required
def upload():
    url = request.form.get('url')
    file = request.files.get('file')

    if url:
        try:
            random_string = ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))
            random_filename = random_string + '.mp4'
            upload_dir = os.path.join(current_app.root_path, 'static', 'uploads')
            file_path = os.path.join(upload_dir, random_filename)
            
            ydl_opts = {
                'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
                'outtmpl': file_path,
                'quiet': True,
                'no_warnings': True,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                title = info.get('title', 'YouTube Video')

            new_video = Videos(
                file_name=random_filename,
                file_extension='.mp4',
                original_filename=title + '.mp4',
                file_path=file_path,
                video_processed=0,
                percent_processed=0,
                posted_by=current_user.name)

            db.session.add(new_video)
            db.session.commit()
            flash('YouTube video has been saved!', 'success')
            return redirect(url_for('main.gallery'))
        except Exception as e:
            flash(f'Error downloading YouTube video: {str(e)}', 'danger')
            return redirect(url_for('main.index'))

    if file and file.filename != '':
        random_string = ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))
        file_extension = os.path.splitext(file.filename)[1]
        random_filename = random_string + file_extension
        file_path = os.path.join(current_app.root_path, 'static', 'uploads', random_filename)

        file.save(file_path)
        new_video = Videos(
            file_name=random_filename,
            file_extension=file_extension,
            original_filename=file.filename,
            file_path=file_path,
            video_processed=0,
            percent_processed=0,
            posted_by=current_user.name)

        db.session.add(new_video)
        db.session.commit()
        flash('Post has been saved!', 'success')
        return redirect(url_for('main.gallery'))

    flash('No file or URL provided', 'warning')
    return redirect(url_for('main.index'))


@main.route('/gallery/<int:id>/download', methods=['GET'])
@login_required
def download_post(id):
    video = Videos.query.get_or_404(id)
    if video.posted_by != current_user.name:
        abort(403) 
    
    filename = video.file_name
    original_filename = video.original_filename
    uploads = os.path.join(current_app.root_path, "static/uploads/")
#    return send_from_directory(uploads, filename=filename, as_attachment=True, download_name=original_filename)
    return send_from_directory(uploads, filename, as_attachment=True,download_name=original_filename)


@main.route('/gallery')
@login_required
def gallery():
    videos = Videos.query.all()
    return render_template('gallery.html', videos=videos , active='gallery')

@main.route('/gallery/<int:id>/delete', methods=['GET'])
@login_required
def delete_post(id):
    video = Videos.query.get_or_404(id)
    if video.posted_by != current_user.name:
        abort(403) 
    db.session.delete(video)
    db.session.commit()
    flash('Post has been deleted!','danger')
    return redirect(url_for('main.gallery'))

@main.route('/gallery/<int:id>/translate', methods=['GET','POST'])
@login_required
def translate_post(id):
    if request.method == 'GET':
        video = Videos.query.get_or_404(id)
        if video.posted_by != current_user.name:
            abort(403)
        flash('Post can be translated!','success')
        return render_template('translate_post.html', video=video)

    elif request.method == 'POST':
        video = Videos.query.get_or_404(id)
        if video.posted_by != current_user.name:
            abort(403)
        flash('Post can be translated!','success')

        #url = request.form["url"]
        filepath = video.file_path
        language_voice = request.form.get('translateTo')
        gender_voice = request.form.get('gender')
        if gender_voice == "male":
            gender = 'male'
        else :
            gender = 'female'

        print(f'File Uploaded : {filepath}')
        print(f'Translate To : {language_voice}')
        print(f'Voice Gender : {gender_voice}')

        random_string = ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))
        file_extension = video.file_extension
        random_filename = random_string+file_extension
        file_path = os.path.join(current_app.root_path, 'static', 'uploads', random_filename)


        new_video = Videos(
            file_name=random_filename,
            file_extension=file_extension,
            original_filename=video.original_filename,
            file_path=file_path, 
            video_processed=1,
            percent_processed=100,
            posted_by=current_user.name)

        db.session.add(new_video)
        db.session.commit()


        random_string = ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))
        file_extension = video.file_extension
        random_filename = random_string+file_extension
        output_path = os.path.join(current_app.root_path, 'static', 'uploads', random_filename)


        try:
            translate_video(filepath,language_voice,gender_voice, output_path)
            flash('Video Translated Succesfully','success')
        except Exception as e:
            flash(f'Translation failed: {str(e)}', 'danger')
            return redirect(url_for('main.gallery'))
#        uploads = os.path.join(current_app.root_path, "../")
#        filename='output.mp4'
#    return send_from_directory(uploads, filename=filename, as_attachment=True, download_name=original_filename)
#        return send_from_directory(uploads, filename, as_attachment=True,download_name=video.original_filename)
        filename = random_filename
        original_filename = video.original_filename
        uploads = os.path.join(current_app.root_path, "static/uploads/")
#    return send_from_directory(uploads, filename=filename, as_attachment=True, download_name=original_filename)
        return send_from_directory(uploads, filename, as_attachment=True,download_name=original_filename)

        #return send_file('output.mp4',   download_name=(f'Dubbed_{language_voice}_{gender_voice}_{video.original_filename}.mp4') ,as_attachment=True)

        if url:
            youtube = YouTube(url)
            filename_original = youtube.title
            video = youtube.streams.get_highest_resolution()
            random_string = ''.join(random.choices(string.ascii_lowercase + string.digits, k=10))
            filename = random_string + '.mp4'
            video_path = video.download(filename=filename)
            print(f'Filename : {filename}')
            print(f'Filepath : {video_path}')
            flash('File Uploaded','success')
            print(video_path)
            translate_video(video_path,language_voice,gender_voice, random_string)
            flash('Video Translated Succesfully','success')
            return send_file('output.mp4',   download_name=(f'Dubbed_{language_voice}_{gender_voice}_{filename_original}.mp4') ,as_attachment=True)


        else:
            flash('File not allowed. Only mp4, avi and mkv are allowed','warning')
            return redirect('/')
    return redirect('/')


#translate_video('/home/shubhankar/Test/flask_auth_app/project/static/uploads/ruxhnff711.mp4','hindi','male')

